from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import os
import time
import json
import uuid
import random
import tempfile
import threading
import queue

import faster_whisper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250MB max file size

MAX_PARALLEL_JOBS = 3
MAX_QUEUED_JOBS = 5

# faster_whisper model
fw = faster_whisper.WhisperModel('ivrit-ai/faster-whisper-v2-d3-e3')

# Dictionary to store temporary file paths
temp_files = {}

# Queue for managing jobs
job_queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)

# Set to keep track of currently running jobs
running_jobs = set()

# Lock for thread-safe operations
lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        
        try:
            file.save(temp_file_path)
        except Exception as e:
            return jsonify({'error': f'File upload failed: {str(e)}'}), 500
        
        job_id = str(uuid.uuid4())
        
        # Store the temporary file path
        temp_files[job_id] = temp_file_path
        
        # Try to add the job to the queue
        try:
            job_queue.put_nowait(job_id)
            return jsonify({'job_id': job_id, 'queued': job_queue.qsize()})
        except queue.Full:
            cleanup_temp_file(job_id)
            return jsonify({'error': 'Server is busy. Please try again later.'}), 503
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/stream/<job_id>')
def stream(job_id):
    if job_id not in temp_files:
        return jsonify({'error': 'Invalid job ID'}), 400

    def generate():
        queue_position = list(job_queue.queue).index(job_id) + 1 if job_id in job_queue.queue else 0
        yield f"data: {json.dumps({'queue_position': queue_position})}\n\n"

        while job_id in job_queue.queue:
            time.sleep(1)
            queue_position = list(job_queue.queue).index(job_id) + 1
            yield f"data: {json.dumps({'queue_position': queue_position})}\n\n"

            process_queue()

        temp_file_path = temp_files[job_id]

        try:
            segs, _ = fw.transcribe(temp_file_path, language='he')

            for seg in segs:
                segment = {
                    'id': seg.id,
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'avg_logprob': seg.avg_logprob,
                    'compression_ratio': seg.compression_ratio,
                    'no_speech_prob': seg.no_speech_prob
                }
                
                current_time = seg.end
                
                data = json.dumps({
                    'progress': current_time / 260,
                    'segment': segment
                })
                yield f"data: {data}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Clean up resources
            cleanup_temp_file(job_id)
            with lock:
                running_jobs.remove(job_id)

    return Response(generate(), content_type='text/event-stream')

def cleanup_temp_file(job_id):
    if job_id in temp_files:
        temp_file_path = temp_files[job_id]
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Error deleting temporary file: {str(e)}")
        finally:
            del temp_files[job_id]

def process_queue():
    with lock:
        if len(running_jobs) >= MAX_PARALLEL_JOBS:
            return
        
        if job_queue.empty():
            return
        
        job_id = job_queue.get()
        running_jobs.add(job_id)

def process_job(job_id):
    # This function would typically start the actual transcription process
    # For now, it's just a placeholder
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500, ssl_context=('secrets/fullchain.pem', 'secrets/privkey.pem'), debug=True)

