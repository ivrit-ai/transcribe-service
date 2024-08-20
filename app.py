from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import os
import time
import json
import uuid
import random  # For generating mock data

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": f"File upload failed: {str(e)}"}), 500

        job_id = str(uuid.uuid4())

        return jsonify({"job_id": job_id})

    return jsonify({"error": "Invalid file"}), 400


@app.route("/stream/<job_id>")
def stream(job_id):
    def generate():
        # Simulate a transcription process with segments
        words = "זוהי הדגמה של תמלול בזמן אמת עבור קובץ השמע שהועלה. אנו מדמים כאן תהליך תמלול ארוך יותר כדי להדגים את הזרימה בזמן אמת.".split()
        total_segments = len(words) // 3 + (1 if len(words) % 3 else 0)
        current_time = 0

        for i in range(total_segments):
            time.sleep(0.5)  # Simulate processing time

            segment_words = words[i * 3 : i * 3 + 3]
            segment_text = " ".join(segment_words)
            segment_duration = sum(len(word) * 0.1 for word in segment_words)

            segment = {
                "id": i,
                "start": current_time,
                "end": current_time + segment_duration,
                "text": segment_text,
                "avg_logprob": random.uniform(-1, 0),
                "compression_ratio": random.uniform(0.8, 1.2),
                "no_speech_prob": random.uniform(0, 0.1),
            }

            current_time += segment_duration

            data = json.dumps({"progress": (i + 1) / total_segments, "segment": segment})
            yield f"data: {data}\n\n"

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
