from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
    redirect,
    url_for,
    session,
    stream_with_context,
    send_file,
)
from flask_oauthlib.client import OAuth
from werkzeug.utils import secure_filename
import requests
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
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from urllib.parse import urljoin
from functools import wraps

import traceback

import posthog
import ffmpeg
import base64
import argparse

dotenv.load_dotenv()

# Parse CLI arguments for rate limiting configuration
parser = argparse.ArgumentParser(description='Transcription service with rate limiting')
parser.add_argument('--max-minutes-per-week', type=int, default=420, help='Maximum credit grant in minutes per week (default: 420)')
args, unknown = parser.parse_known_args()

# Rate limiting configuration from CLI arguments
MAX_MINUTES_PER_WEEK = args.max_minutes_per_week  # Maximum credit grant per week
REPLENISH_RATE_MINUTES_PER_DAY = MAX_MINUTES_PER_WEEK / 7  # Automatically derive daily replenish rate

# RunPod configuration
RUNPOD_MAX_PAYLOAD_LEN = 10 * 1024 * 1024  # 10MB max payload length

in_dev = "TS_STAGING_MODE" in os.environ
in_hiatus_mode = "TS_HIATUS_MODE" in os.environ

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max file size
app.secret_key = os.environ["FLASK_APP_SECRET"]

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure file logging
file_handler = RotatingFileHandler(
    filename="app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(file_handler)

verbose = "VERBOSE" in os.environ


def log_message_in_session(message):
    user_email = session.get("user_email")
    if user_email:
        logger.info(f"{user_email}: {message}")
    else:
        logger.info(f"{message}")


def log_message(message):
    logger.info(f"{message}")


# Configure Google OAuth
oauth = OAuth(app)
google = oauth.remote_app(
    "google",
    consumer_key=os.environ["GOOGLE_CLIENT_ID"],
    consumer_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    request_token_params={"scope": "email"},
    base_url="https://www.googleapis.com/oauth2/v1/",
    request_token_url=None,
    access_token_method="POST",
    access_token_url="https://accounts.google.com/o/oauth2/token",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
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


MAX_PARALLEL_SHORT_JOBS = 1
MAX_PARALLEL_LONG_JOBS = 1
MAX_QUEUED_JOBS = 20
SHORT_JOB_THRESHOLD = 20 * 60
JOB_TIMEOUT = 1 * 60

# Dictionary to store temporary file paths
temp_files = {}

# Queues for managing jobs
short_job_queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)
long_job_queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)

# Dictionaries to keep track of currently running jobs
running_short_jobs = {}
running_long_jobs = {}

# Keep track of job results
job_results = {}

# Lock for thread-safe operations
lock = threading.RLock()

# Dictionary to keep track of user's active jobs
user_jobs = {}

# Dictionary to keep track of last access time for each job
job_last_accessed = {}

# Dictionary to keep track of job threads
job_threads = {}

# Dictionary to store user rate limiting buckets
user_buckets = {}

class RunPodJob:
    def __init__(self, api_key: str, endpoint_id: str, payload: dict):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Submit the job immediately on creation
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        self.job_id = result.get("id")
    
    def status(self):
        """Get job status"""
        response = requests.get(
            f"{self.base_url}/status/{self.job_id}",
            headers=self.headers
        )
        response.raise_for_status()

        status_response = response.json()
        return status_response.get("status", "UNKNOWN")
    
    def stream(self):
        """Stream job results"""
        while True:
            response = requests.get(
                f"{self.base_url}/stream/{self.job_id}",
                headers=self.headers,
                stream=True
            )
            response.raise_for_status()
            
            # Expect a single response        
            try:
                content = response.content.decode('utf-8')
                data = json.loads(content)
                if data['status'] != 'IN_PROGRESS':
                    break

                for item in data['stream']:
                    yield item['output']
            except json.JSONDecodeError as e:
                log_message(f"Failed to parse JSON response: {e}")
                return
    
    def cancel(self):
        """Cancel the job"""
        response = requests.post(
            f"{self.base_url}/cancel/{self.job_id}",
            headers=self.headers
        )
        response.raise_for_status()

        return response.json()

class LeakyBucket:
    def __init__(self, max_minutes_per_week):
        self.max_seconds = max_minutes_per_week * 60  # Convert to seconds
        self.seconds_remaining = max_minutes_per_week * 60
        self.last_update = time.time()

        # Calculate fill rate (per second) based on daily replenish rate
        self.time_fill_rate = (REPLENISH_RATE_MINUTES_PER_DAY * 60) / (24 * 3600)  # Convert daily rate to per-second

    def update(self):
        """Update bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Add resources based on fill rate
        self.seconds_remaining = min(self.max_seconds, self.seconds_remaining + self.time_fill_rate * elapsed)

    def can_transcribe(self, duration_seconds):
        """Check if transcription is allowed."""
        self.update()
        return self.seconds_remaining >= duration_seconds

    def consume(self, duration_seconds):
        """Consume resources for transcription."""
        self.update()
        self.seconds_remaining -= duration_seconds
        return self.seconds_remaining > 0

    def get_remaining_minutes(self):
        """Get remaining minutes in the bucket."""
        self.update()
        return self.seconds_remaining / 60

    def get_remaining_seconds(self):
        """Get remaining seconds in the bucket."""
        self.update()
        return self.seconds_remaining

    def is_fully_replenished(self):
        """Check if the bucket is fully replenished."""
        self.update()
        return self.seconds_remaining >= self.max_seconds

ffmpeg_supported_mimes = [
    "video/",
    "audio/",
    "application/mp4",
    "application/x-matroska",
    "application/mxf",
]


def is_ffmpeg_supported_mimetype(file_mime):
    return any(file_mime.startswith(supported_mime) for supported_mime in ffmpeg_supported_mimes)


def get_media_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        audio_info = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        return float(audio_info["duration"])
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")
        return None


def get_user_quota(user_email):
    """Get or create a rate limiting bucket for a user."""
    if user_email not in user_buckets:
        user_buckets[user_email] = LeakyBucket(MAX_MINUTES_PER_WEEK)
    return user_buckets[user_email]


SPEEDUP_FACTOR = 20


def calculate_queue_time(queue_to_use, running_jobs, exclude_last=False):
    """
    Calculate the estimated time remaining for jobs in queue and running jobs

    Args:
        queue_to_use: The queue to check (short_job_queue or long_job_queue)
        running_jobs: Dictionary of currently running jobs
        exclude_last: Whether to exclude the last job in queue from calculation

    Returns:
        time_ahead_str: Formatted string of estimated time (HH:MM:SS)
    """
    with lock:
        time_ahead = 0

        # Add remaining time of currently running jobs
        for running_job_id, running_job in running_jobs.items():
            if running_job_id in job_results:
                # Get progress of the running job
                progress = job_results[running_job_id]["progress"]
                # Add only the remaining duration
                remaining_duration = running_job.duration * (1 - progress)
                time_ahead += remaining_duration
            else:
                # If no progress info yet, add full duration
                time_ahead += running_job.duration

        # Add duration of queued jobs
        queue_jobs = list(queue_to_use.queue)
        if exclude_last and queue_jobs:
            queue_jobs = queue_jobs[:-1]  # Exclude the last job

        time_ahead += sum(job.duration for job in queue_jobs)

        # Apply speedup factor
        time_ahead /= SPEEDUP_FACTOR

        # Convert time_ahead to HH:MM:SS format
        return str(timedelta(seconds=int(time_ahead)))


def queue_job(job_id, user_email, filename, duration):
    # Try to add the job to the queue
    log_message_in_session(f"Queuing job {job_id}...")

    with lock:
        # Check if user already has a job queued or running
        if user_email in user_jobs:
            return False, (
                jsonify({"error": "יש לך כבר עבודה בתור או בביצוע. אנא המתן לסיומה לפני העלאת קובץ חדש."}),
                200,
            )

        # Check rate limits
        user_bucket = get_user_quota(user_email)
        if not user_bucket.can_transcribe(duration):
            remaining_minutes = user_bucket.get_remaining_minutes()
            log_message_in_session(f"Job queuing rate limited for user {user_email}. Requested: {duration/60:.1f}min, Remaining: {remaining_minutes:.1f}min")
            
            # Calculate how many minutes they need to wait, accounting for replenish rate
            needed_minutes = (duration / 60) - remaining_minutes
            replenish_rate_per_minute = user_bucket.time_fill_rate * 60
            wait_minutes = needed_minutes / (1 + replenish_rate_per_minute)

            # Convert time_fill_rate (per second) to minutes per minute
            error_msg = f"אנא המתן {wait_minutes:.1f} דקות לפני העלאת קובץ חדש."
            
            return False, (jsonify({"error": error_msg}), 429)

        try:
            job_desc = box.Box()
            job_desc.qtime = time.time()
            job_desc.utime = time.time()
            job_desc.id = job_id
            job_desc.user_email = user_email
            job_desc.filename = filename
            job_desc.duration = duration

            if duration <= SHORT_JOB_THRESHOLD:
                queue_to_use = short_job_queue
                job_type = "short"
                running_jobs = running_short_jobs
            else:
                queue_to_use = long_job_queue
                job_type = "long"
                running_jobs = running_long_jobs

            job_desc.job_type = job_type

            queue_depth = queue_to_use.qsize()
            queue_to_use.put_nowait(job_desc)

            # Calculate time ahead only for the relevant queue
            time_ahead_str = calculate_queue_time(queue_to_use, running_jobs, exclude_last=True)

            # Add job to user's active jobs
            user_jobs[user_email] = job_id

            # Set initial access time
            job_last_accessed[job_id] = time.time()

            capture_event(job_id, "job-queued", {"user": user_email, "queue-depth": queue_depth, "job-type": job_type})

            log_message_in_session(
                f"Job queued successfully: {job_id}, queue depth: {queue_depth}, job type: {job_type}, job desc: {job_desc}"
            )

            return True, (
                jsonify({"job_id": job_id, "queued": queue_depth, "job_type": job_type, "time_ahead": time_ahead_str})
            )
        except queue.Full:
            capture_event(job_id, "job-queue-failed", {"queue-depth": queue_depth, "job-type": job_type})

            log_message_in_session(f"Job queuing failed: {job_id}")

            cleanup_temp_file(job_id)
            return False, (jsonify({"error": "השרת עמוס כרגע. אנא נסה שוב מאוחר יותר."}), 503)


@app.route("/")
def index():
    if in_dev:
        session["user_email"] = os.environ["TS_USER_EMAIL"]

    if "user_email" not in session:
        return redirect(url_for("login"))

    if in_hiatus_mode:
        return render_template("server-down.html")
    
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html", google_analytics_tag=os.environ["GOOGLE_ANALYTICS_TAG"])


@app.route("/authorize")
def authorize():
    return google.authorize(callback=url_for("authorized", _external=True))


@app.route("/login/authorized")
def authorized():
    resp = google.authorized_response()
    if resp is None or resp.get("access_token") is None:
        error_message = "Access denied: reason={0} error={1}".format(
            request.args.get("error_reason", "Unknown"), request.args.get("error_description", "Unknown")
        )
        return render_template("close_window.html", success=False, message=error_message)

    session["google_token"] = (resp["access_token"], "")
    user_info = google.get("userinfo")
    session["user_email"] = user_info.data["email"]

    session.pop("google_token")

    return render_template("close_window.html", success=True)


@google.tokengetter
def get_google_oauth_token():
    return session.get("google_token")


@app.route("/upload", methods=["POST"])
def upload_file():
    job_id = str(uuid.uuid4())
    user_email = session.get("user_email")

    if in_hiatus_mode:
        capture_event(job_id, "file-upload-hiatus-rejected", {"user": user_email})
        return jsonify({"error": "השירות כרגע לא פעיל. אנא נסה שוב מאוחר יותר."}), 503

    capture_event(job_id, "file-upload", {"user": user_email})

    if "file" not in request.files:
        return jsonify({"error": "לא נבחר קובץ. אנא בחר קובץ להעלאה."}), 200

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "שם הקובץ ריק. אנא בחר קובץ תקין."}), 200

    if not file:
        return jsonify({"error": "הקובץ שנבחר אינו תקין. אנא נסה קובץ אחר."}), 200

    filename = secure_filename(file.filename)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    try:
        file.save(temp_file_path)
    except Exception as e:
        return jsonify({"error": f"העלאת הקובץ נכשלה: {str(e)}"}), 200

    # Get the MIME type of the file
    filetype = magic.Magic(mime=True).from_file(temp_file_path)

    if not is_ffmpeg_supported_mimetype(filetype):
        return jsonify({"error": f"סוג הקובץ {filetype} אינו נתמך. אנא העלה קובץ אודיו או וידאו תקין."}), 200

    # Get the duration of the audio file
    duration = get_media_duration(temp_file_path)

    if duration is None:
        return jsonify({"error": "לא ניתן לקרוא את משך הקובץ. אנא ודא שהקובץ תקין ונסה שוב."}), 200

    # Store the temporary file path
    with lock:
        temp_files[job_id] = temp_file_path

        queued, res = queue_job(job_id, user_email, filename, duration)
        if queued:
            job_results[job_id] = {"results": [], "completion_time": None, "progress": 0}
        else:
            cleanup_temp_file(job_id)

    return res


@app.route("/job_status/<job_id>")
def job_status(job_id):
    # Make sure the job has been queued
    if job_id not in job_results:
        return jsonify({"error": "מזהה העבודה אינו תקין. אנא נסה להעלות את הקובץ מחדש."}), 400

    with lock:
        # Update last access time for the job
        job_last_accessed[job_id] = time.time()

        # Check if the job is in the queue
        queue_position = None
        job_type = None
        queue_to_use = None

        # Check short queue
        for idx, job_desc in enumerate(short_job_queue.queue):
            if job_desc.id == job_id:
                queue_position = idx + 1
                job_type = "short"
                queue_to_use = short_job_queue
                break

        # If not found in short queue, check long queue
        if not queue_position:
            for idx, job_desc in enumerate(long_job_queue.queue):
                if job_desc.id == job_id:
                    queue_position = idx + 1
                    job_type = "long"
                    queue_to_use = long_job_queue
                    break

        if queue_position:
            running_jobs = running_short_jobs if job_type == "short" else running_long_jobs
            time_ahead_str = calculate_queue_time(queue_to_use, running_jobs)
            return jsonify({"queue_position": queue_position, "time_ahead": time_ahead_str, "job_type": job_type})

        # If job is in progress, return only the progress
        if job_results[job_id]["progress"] < 1.0:
            return jsonify({"progress": job_results[job_id]["progress"]})

        # If job is complete, return the full job_result data
        return jsonify(job_results[job_id])


def cleanup_temp_file(job_id):
    if job_id in temp_files:
        temp_file_path = temp_files[job_id]
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            log_message(f"Error deleting temporary file: {str(e)}")
        finally:
            del temp_files[job_id]


def clean_some_unicode_from_text(text):
    chars_to_remove = "\u061C"  # Arabic letter mark
    chars_to_remove += "\u200B\u200C\u200D"  # Zero-width space, non-joiner, joiner
    chars_to_remove += "\u200E\u200F"  # LTR and RTL marks
    chars_to_remove += "\u202A\u202B\u202C\u202D\u202E"  # LTR/RTL embedding, pop, override
    chars_to_remove += "\u2066\u2067\u2068\u2069"  # Isolate controls
    chars_to_remove += "\uFEFF"  # Zero-width no-break space

    return text.translate({ord(c): None for c in chars_to_remove})


@app.route("/download/<job_id>", methods=["GET"])
def download_file(job_id):
    if job_id not in temp_files:
        return jsonify({"error": "File not found"}), 404

    return send_file(temp_files[job_id])





def process_segment(job_id, segment, duration):
    """Process a single segment and update job results"""
    with lock:
        # Check if the job should be terminated
        if time.time() - job_last_accessed.get(job_id, 0) > JOB_TIMEOUT:
            log_message(f"Terminating inactive job: {job_id}")
            return False

        segment_data = {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": clean_some_unicode_from_text(segment["text"]),
            "avg_logprob": segment["avg_logprob"],
            "compression_ratio": segment["compression_ratio"],
            "no_speech_prob": segment["no_speech_prob"],
        }

        progress = segment["end"] / duration
        job_results[job_id]["progress"] = progress
        job_results[job_id]["results"].append(segment_data)

    return True


def transcribe_job(job_desc):
    job_id = job_desc.id

    run_request = None

    try:
        log_message(f"{job_desc.user_email}: beginning transcription of {job_desc}, file name={job_desc.filename}")

        temp_file_path = temp_files[job_id]
        duration = job_desc.duration

        # Consume quota immediately when transcription starts
        user_bucket = get_user_quota(job_desc.user_email)
        user_bucket.consume(duration)
        remaining_minutes = user_bucket.get_remaining_minutes()
        log_message(f"{job_desc.user_email}: consumed {duration/60:.1f} minutes from quota. Remaining: {remaining_minutes:.1f} minutes")

        transcribe_start_time = time.time()
        capture_event(
            job_id,
            "transcribe-start",
            {"user": job_desc.user_email, "queued_seconds": transcribe_start_time - job_desc.qtime, "job-type": job_desc.job_type},
        )

        log_message(f"{job_desc.user_email}: job {job_desc} has a duration of {duration} seconds")

        # Prepare and send request to runpod
        if in_dev:
            # In dev mode, send file as blob
            audio_data = open(temp_file_path, 'rb').read()
            payload = {
                "input": {
                    "type": "blob",
                    "data": base64.b64encode(audio_data).decode('utf-8'),
                    "model": "ivrit-ai/whisper-large-v3-turbo-ct2",
                    "streaming": True
                }
            }
            
            # Check payload size
            if len(str(payload)) > RUNPOD_MAX_PAYLOAD_LEN:
                log_message(f"Payload length is {len(str(payload))}, exceeding max payload length of {RUNPOD_MAX_PAYLOAD_LEN}.")
                return
        else:
            # In production mode, send file as URL
            base_url = os.environ["BASE_URL"]
            download_url = urljoin(base_url, f"/download/{job_id}")
            payload = {
                "input": {"type": "url", "url": download_url, "model": "ivrit-ai/whisper-large-v3-turbo-ct2", "streaming": True}
            }

        # Start streaming request using RunPodJob
        run_request = RunPodJob(os.environ["RUNPOD_API_KEY"], os.environ["RUNPOD_ENDPOINT_ID"], payload)

        for i in range(30):
            if run_request.status() == "IN_QUEUE":
                time.sleep(10)
                continue

            break

        # Process streaming results
        timeouts = 0
        while True:
            try:
                for segment in run_request.stream():
                    if "error" in segment:
                        log_message(f"Error in RunPod transcription stream: {segment['error']}")
                        return

                    # Process segment
                    if not process_segment(job_id, segment, duration):
                        # Job was terminated
                        log_message(f"Terminating runpod job due to process_segment error.")
                        run_request.cancel()
                        return

                run_request = None

                break

            except requests.exceptions.ReadTimeout as e:
                log_message(f"run_request.stream() time out #{timeouts}, trying again.")
                timeouts += 1
                pass

            except Exception as e:
                log_message(f"Exception during run_request.stream(): {e}")
                print(traceback.format_exc())
                raise e

        log_message(f"{job_desc.user_email}: done transcribing job {job_id}, audio duration was {duration}.")

        transcribe_done_time = time.time()
        capture_event(
            job_id,
            "transcribe-done",
            {
                "user": job_desc.user_email,
                "transcription_seconds": transcribe_done_time - transcribe_start_time,
                "audio_duration_seconds": duration,
                "job-type": job_desc.job_type,
            },
        )

        with lock:
            job_results[job_id]["progress"] = 1.0
            job_results[job_id]["completion_time"] = datetime.now()
    except Exception as e:
        log_message(f"Error in transcription job {job_id}: {str(e)}")
    finally:
        if run_request:
            try:
                log_message("Terminating run_request at end of transcribe_job::finally")
                run_request.cancel()
            except Exception as e:
                log_message(f"Failed to terminate runpod job")

        if job_desc.duration <= SHORT_JOB_THRESHOLD:
            del running_short_jobs[job_id]
        else:
            del running_long_jobs[job_id]
        # Remove job from user's active jobs
        with lock:
            user_email = job_desc.user_email
            if user_email in user_jobs and user_jobs[user_email] == job_id:
                del user_jobs[user_email]
        cleanup_temp_file(job_id)
        if job_id in job_threads:
            del job_threads[job_id]


def submit_next_task(job_queue, running_jobs, max_parallel_jobs):
    with lock:
        if len(running_jobs) < max_parallel_jobs:
            if not job_queue.empty():
                job_desc = job_queue.get()
                running_jobs[job_desc.id] = job_desc
                t_thread = threading.Thread(target=transcribe_job, args=(job_desc,))
                job_threads[job_desc.id] = t_thread
                t_thread.start()


def cleanup_old_results():
    with lock:
        current_time = datetime.now()
        jobs_to_delete = []
        for job_id, job_data in job_results.items():
            if job_data["completion_time"] and (current_time - job_data["completion_time"]) > timedelta(minutes=30):
                jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            del job_results[job_id]

        # Clean up user buckets that are fully replenished
        users_to_remove = []
        for user_email, bucket in user_buckets.items():
            # Remove bucket if it is fully replenished
            if bucket.is_fully_replenished():
                users_to_remove.append(user_email)
        
        for user_email in users_to_remove:
            del user_buckets[user_email]


def terminate_inactive_jobs():
    with lock:
        current_time = time.time()
        jobs_to_terminate = []

        # Check queued jobs.
        # Running jobs are handled in transcribe_job.
        for queue_to_check in [short_job_queue, long_job_queue]:
            for job in list(queue_to_check.queue):
                if current_time - job_last_accessed.get(job.id, 0) > JOB_TIMEOUT:
                    jobs_to_terminate.append(job.id)
                    queue_to_check.queue.remove(job)

        # Terminate and clean up jobs
        for job_id in jobs_to_terminate:
            log_message(f"Terminating inactive job: {job_id}")
            if job_id in job_results:
                del job_results[job_id]
            if job_id in job_last_accessed:
                del job_last_accessed[job_id]
            cleanup_temp_file(job_id)

            # Remove job from user's active jobs
            for user_email, active_job_id in list(user_jobs.items()):
                if active_job_id == job_id:
                    del user_jobs[user_email]

            # The job thread will terminate itself in the next iteration of the transcribe_job function


def event_loop():
    while True:
        submit_next_task(short_job_queue, running_short_jobs, MAX_PARALLEL_SHORT_JOBS)
        submit_next_task(long_job_queue, running_long_jobs, MAX_PARALLEL_LONG_JOBS)
        cleanup_old_results()
        terminate_inactive_jobs()
        time.sleep(0.1)


el_thread = threading.Thread(target=event_loop)
el_thread.start()

if __name__ == "__main__":
    port = 4600 if in_dev else 4500
    app.run(host="0.0.0.0", port=port, ssl_context=("secrets/fullchain.pem", "secrets/privkey.pem"))
