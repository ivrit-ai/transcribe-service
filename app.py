from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, session
from flask_oauthlib.client import OAuth
from werkzeug.utils import secure_filename
import os
import time
import json
import uuid
import box
import dotenv
import magic
import random
import tempfile
import threading
import queue
import logging
from datetime import datetime, timedelta

import faster_whisper
import posthog
import pydub

dotenv.load_dotenv()

in_dev = 'TS_STAGING_MODE' in os.environ

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB max file size
app.secret_key = os.environ['FLASK_APP_SECRET']

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

verbose = "VERBOSE" in os.environ

def log_message_in_session(message):
    user_email = session.get('user_email')
    if user_email: 
        logger.info(f"{user_email}: {message}")
    else:
        logger.info(f"{message}")

def log_message(message):
    logger.info(f"{message}")


# Configure Google OAuth
oauth = OAuth(app)
google = oauth.remote_app(
    'google',
    consumer_key=os.environ['GOOGLE_CLIENT_ID'],
    consumer_secret=os.environ['GOOGLE_CLIENT_SECRET'],
    request_token_params={
        'scope': 'email'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

ph = None
if "POSTHOG_API_KEY" in os.environ:
    ph = posthog.Posthog(project_api_key=os.environ["POSTHOG_API_KEY"], host="https://us.i.posthog.com")

def capture_event(distinct_id, event, props=None):
    global ph

    if not ph:
        return

    props = {} if not props else props
    props["source"] = "transcribe.ivrit.ai"

    ph.capture(distinct_id=distinct_id, event=event, properties=props)

MAX_PARALLEL_JOBS = 3
MAX_QUEUED_JOBS = 10

# faster_whisper model
fw = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")

# Dictionary to store temporary file paths
temp_files = {}

# Queue for managing jobs
job_queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)

# Set to keep track of currently running jobs
running_jobs = {}

# Keep track of job results
job_results = {}

# Lock for thread-safe operations
lock = threading.Lock()

ffmpeg_supported_mimes = [
    "video/",
    "audio/",
    "application/mp4",
    "application/x-matroska",
    "application/mxf",
]

def is_ffmpeg_supported_mimetype(file_mime):
    return any(file_mime.startswith(supported_mime) for supported_mime in ffmpeg_supported_mimes)

def queue_job(job_id, user_email, filename):
    # Try to add the job to the queue
    log_message_in_session(f"Queuing job {job_id}...")

    try:
        job_desc = box.Box()
        job_desc.qtime = time.time()
        job_desc.utime = time.time()
        job_desc.id = job_id
        job_desc.user_email = user_email
        job_desc.filename = filename

        queue_depth = job_queue.qsize()
        job_queue.put_nowait(job_desc)

        capture_event(job_id, "job-queued", {"queue-depth": queue_depth})

        log_message_in_session(f"Job queued successfully: {job_id}, queue depth: {queue_depth}")

        return True, (jsonify({"job_id": job_id, "queued": job_queue.qsize()}))
    except queue.Full:
        capture_event(job_id, "job-queue-failed", {"queue-depth": job_queue.qsize()})

        log_message_in_session(f"Job queuing failed: {job_id}")

        cleanup_temp_file(job_id)
        return False, (jsonify({"error": "Server is busy. Please try again later."}), 503)

@app.route("/")
def index():
    if in_dev:
        session['user_email'] = os.environ['TS_USER_EMAIL']

    if 'user_email' not in session:
        return redirect(url_for('login'))

    return render_template("index.html")

@app.route('/login')
def login():
    return render_template('login.html',
                           google_analytics_tag=os.environ['GOOGLE_ANALYTICS_TAG'])

@app.route('/authorize')
def authorize():
    return google.authorize(callback=url_for('authorized', _external=True))

@app.route('/login/authorized')
def authorized():
    resp = google.authorized_response()
    if resp is None or resp.get('access_token') is None:
        error_message = 'Access denied: reason={0} error={1}'.format(
            request.args.get('error_reason', 'Unknown'),
            request.args.get('error_description', 'Unknown')
        )
        return render_template('close_window.html', success=False, message=error_message)

    session['google_token'] = (resp['access_token'], '')
    user_info = google.get('userinfo')
    session['user_email'] = user_info.data["email"]

    session.pop('google_token')

    return render_template('close_window.html', success=True)

@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

@app.route("/upload", methods=["POST"])
def upload_file():
    job_id = str(uuid.uuid4())

    capture_event(job_id, "file-upload")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file:
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    try:
        file.save(temp_file_path)
    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

    # Get the MIME type of the file
    filetype = magic.Magic(mime=True).from_file(temp_file_path)

    if not is_ffmpeg_supported_mimetype(filetype):
        return jsonify({"error": f"File uploaded is of type {filetype}, which is unsupported"}), 400

    # Store the temporary file path
    with lock:
        temp_files[job_id] = temp_file_path

        queued, res = queue_job(job_id, session.get('user_email'), filename)
        if queued:
            job_results[job_id] = {
                'results': [],
                'completion_time': None,
                'progress': 0
            }

    return res

@app.route("/job_status/<job_id>")
def job_status(job_id):
    # Make sure the job has been queued
    if job_id not in job_results:
        return jsonify({"error": "Invalid job ID"}), 400

    with lock:
        # Check if the job is in the queue
        queue_position = None
        for idx, job_desc in enumerate(job_queue.queue):
            if job_desc.id == job_id:
                queue_position = idx + 1
                break

        if queue_position:
            return jsonify({'queue_position': queue_position})

        # If job is complete, return the entire job_result data structure
        if job_results[job_id]['progress'] == 1.0:
            return jsonify(job_results[job_id])

        # If job is in progress, return only the progress
        return jsonify({'progress': job_results[job_id]['progress']})

def cleanup_temp_file(job_id):
    if job_id in temp_files:
        temp_file_path = temp_files[job_id]
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            log_message(f"Error deleting temporary file: {str(e)}")
        finally:
            del temp_files[job_id]

def transcribe_job(job_desc):
    job_id = job_desc.id

    try:
        log_message(f"{job_desc.user_email}: beginning transcription of {job_desc}, file name={job_desc.filename}")

        temp_file_path = temp_files[job_id]
        duration = pydub.AudioSegment.from_file(temp_file_path).duration_seconds

        transcribe_start_time = time.time()
        capture_event(job_id, "transcribe-start", {"queued_seconds": transcribe_start_time - job_desc.qtime})

        log_message(f"{job_desc.user_email}: job {job_desc} has a duration of {duration} seconds")

        segs, _ = fw.transcribe(temp_file_path, language="he")

        for seg in segs:
            segment = {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
            }

            progress = seg.end / duration
            job_results[job_id]['progress'] = progress
            job_results[job_id]['results'].append({"progress": progress, "segment": segment})

        log_message(f"{job_desc.user_email}: done transcribing job {job_id}, audio duration was {duration}.")

        log_message(f"{job_desc.user_email}: done transcribing job {job_id}, audio duration was {duration}.")

        transcribe_done_time = time.time()
        capture_event(
            job_id,
            "transcribe-done",
            {"transcription_seconds": transcribe_done_time - transcribe_start_time, "audio_duration_seconds": duration},
        )

        job_results[job_id]['progress'] = 1.0
        job_results[job_id]['results'].append({"progress": 1.0})
        job_results[job_id]['completion_time'] = datetime.now()
    finally:
        del running_jobs[job_id]
        cleanup_temp_file(job_id)

def submit_next_task():
    with lock:
        if len(running_jobs) < MAX_PARALLEL_JOBS and not job_queue.empty():
            job_desc = job_queue.get()
            running_jobs[job_desc.id] = job_desc

            t_thread = threading.Thread(target=transcribe_job, args=(job_desc,))
            t_thread.start()

def cleanup_old_results():
    with lock:
        current_time = datetime.now()
        jobs_to_delete = []
        for job_id, job_data in job_results.items():
            if job_data['completion_time'] and (current_time - job_data['completion_time']) > timedelta(minutes=30):
                jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            del job_results[job_id]

def event_loop():
    while True:
        submit_next_task()
        cleanup_old_results()
        time.sleep(0.1)

el_thread = threading.Thread(target=event_loop)
el_thread.start()

if __name__ == "__main__":

    port = 4600 if in_dev else 4500

    app.run(host="0.0.0.0", port=port, ssl_context=("secrets/fullchain.pem", "secrets/privkey.pem"), debug=True)
