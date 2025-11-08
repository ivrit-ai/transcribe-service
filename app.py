from fastapi import (
    FastAPI,
    Request,
    Response,
    HTTPException,
    Depends,
    Form,
    File,
    UploadFile,
    status
)
from werkzeug.utils import secure_filename
import aiohttp
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
import uvicorn
import os
import math
import time
import json
import uuid
import box
import dotenv
import magic
import random
import tempfile
import asyncio
import queue
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlencode
from functools import wraps
from typing import Optional
from contextlib import asynccontextmanager
import dataclasses

import traceback
import copy
import hashlib
import gzip
import io

import posthog
import ffmpeg
import base64
import argparse
import re
import ivrit
import json as _json
from cachetools import LRUCache

dotenv.load_dotenv()

# Parse CLI arguments for configuration
parser = argparse.ArgumentParser(description='Transcription service with rate limiting')
parser.add_argument('--max-minutes-per-week', type=int, default=180, help='Maximum credit grant in minutes per week (default: 420)')
parser.add_argument('--staging', action='store_true', help='Enable staging mode')
parser.add_argument('--hiatus', action='store_true', help='Enable hiatus mode')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('--dev', action='store_true', help='Enable development mode')
parser.add_argument('--dev-user-email', help='User email for development mode')
parser.add_argument('--config', dest='config_path', required=True, help='Path to configuration JSON defining languages and models')
args, unknown = parser.parse_known_args()

# Rate limiting configuration from CLI arguments
MAX_MINUTES_PER_WEEK = args.max_minutes_per_week  # Maximum credit grant per week
REPLENISH_RATE_MINUTES_PER_DAY = MAX_MINUTES_PER_WEEK / 7  # Automatically derive daily replenish rate



in_dev = args.staging or args.dev
in_hiatus_mode = args.hiatus
verbose = args.verbose

# Load language/model configuration (mandatory, after args are parsed)
CONFIG_PATH = args.config_path
LANG_CONFIG = {}
with open(CONFIG_PATH, "r", encoding="utf-8") as _cfg_f:
    LANG_CONFIG = _json.load(_cfg_f)

# Basic validation of configuration
if "languages" not in LANG_CONFIG or not isinstance(LANG_CONFIG["languages"], dict):
    raise RuntimeError("Invalid configuration: missing 'languages' dictionary")
for lang_key, lang_cfg in LANG_CONFIG["languages"].items():
    if not isinstance(lang_cfg, dict):
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': must be an object")
    if "model" not in lang_cfg or not isinstance(lang_cfg["model"], str) or not lang_cfg["model"].strip():
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'model'")
    if "general_availability" not in lang_cfg or not isinstance(lang_cfg["general_availability"], bool):
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'general_availability' (bool)")
    if "enabled" not in lang_cfg or not isinstance(lang_cfg["enabled"], bool):
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'enabled' (bool)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global queue_locks
    
    # Startup
    # Initialize locks
    queue_locks[SHORT] = asyncio.Lock()
    queue_locks[LONG] = asyncio.Lock()
    queue_locks[PRIVATE] = asyncio.Lock()
    
    # Start background event loop
    asyncio.create_task(event_loop())
    
    yield

# Create FastAPI app
app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure file logging
file_handler = RotatingFileHandler(
    filename="app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(file_handler)

# Templates
templates = Jinja2Templates(directory="templates")

# Session management (simplified for FastAPI)
sessions = {}



def get_session_id(request: Request) -> str:
    """Get or create session ID from request"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id



def get_user_email(request: Request) -> Optional[str]:
    """Get user email from session"""
    session_id = get_session_id(request)
    return sessions.get(session_id, {}).get("user_email")

def set_user_email(request: Request, email: str) -> str:
    """Set user email in session and return session ID"""
    session_id = get_session_id(request)
    if session_id not in sessions:
        sessions[session_id] = {}
    sessions[session_id]["user_email"] = email
    return session_id

from gdrive_utils import (
    refresh_google_access_token,
    get_access_token_from_refresh,
    upload_to_google_appdata,
    update_google_appdata_file,
    download_google_appdata_file_bytes,
    list_google_appdata_files,
    find_google_appdata_file_by_name,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
)

def log_message(message):
    logger.info(f"{message}")

async def download_toc(refresh_token: Optional[str]) -> dict:
    """Download TOC file from Google Drive (gzipped), using cache if available."""
    cache_key = get_toc_cache_key(refresh_token)
    
    # Try to get from cache first
    if cache_key:
        cached_toc = toc_cache.get(cache_key)
        if cached_toc is not None:
            return copy.deepcopy(cached_toc)
    
    # Not in cache, download from Google Drive
    file_id = await find_google_appdata_file_by_name(refresh_token, "toc.json.gz")
    
    if not file_id:
        toc_data = {"entries": []}
    else:
        # Download raw bytes and decompress
        file_bytes = await download_google_appdata_file_bytes(refresh_token, file_id)
        if not file_bytes:
            toc_data = {"entries": []}
        else:
            file_bytes = gzip.decompress(file_bytes)
            toc_data = json.loads(file_bytes)
    
    # Store in cache (only persistent TOC from Google Drive)
    if cache_key:
        toc_cache[cache_key] = copy.deepcopy(toc_data)
    
    return toc_data

async def upload_toc(refresh_token: Optional[str], toc_data: dict, user_email: Optional[str] = None) -> bool:
    """Upload TOC file (gzipped), updating existing one atomically or creating new one."""
    # Find existing toc.json.gz
    existing_id = await find_google_appdata_file_by_name(refresh_token, "toc.json.gz")
    
    # Prepare data: compress JSON
    json_data = json.dumps(toc_data).encode('utf-8')
    file_data = gzip.compress(json_data)
    mime_type = "application/gzip"
    
    success = False
    if existing_id:
        # Update existing file atomically
        success = await update_google_appdata_file(refresh_token, existing_id, file_data, mime_type, user_email)
    else:
        # Create new file (gzipped)
        success = await upload_to_google_appdata(refresh_token, "toc.json.gz", file_data, mime_type, user_email)
    
    # Invalidate cache for this refresh_token if upload was successful
    if success:
        cache_key = get_toc_cache_key(refresh_token)
        if cache_key:
            toc_cache.pop(cache_key, None)
    
    return success

def get_toc_lock(user_email: str) -> asyncio.Lock:
    """Get or create a lock for a specific user's TOC updates."""
    if user_email not in toc_locks:
        toc_locks[user_email] = asyncio.Lock()
    return toc_locks[user_email]

# PostHog configuration
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

# Queue type constants
SHORT = "short"
LONG = "long"
PRIVATE = "private"

MAX_PARALLEL_SHORT_JOBS = 1
MAX_PARALLEL_LONG_JOBS = 1
MAX_PARALLEL_PRIVATE_JOBS = 1000
MAX_QUEUED_JOBS = 20
MAX_QUEUED_PRIVATE_JOBS = 5000
SHORT_JOB_THRESHOLD = 20 * 60

SPEEDUP_FACTOR = 15
MAX_AUDIO_DURATION_IN_HOURS = 20
EXECUTION_TIMEOUT_MS = int(MAX_AUDIO_DURATION_IN_HOURS * 3600 * 1000 / SPEEDUP_FACTOR)

# Dictionary to store temporary file paths
temp_files = {}

# Programmatic queue management
queues = {
    SHORT: queue.Queue(maxsize=MAX_QUEUED_JOBS),
    LONG: queue.Queue(maxsize=MAX_QUEUED_JOBS),
    PRIVATE: queue.Queue(maxsize=MAX_QUEUED_PRIVATE_JOBS)
}

# Programmatic running jobs management
running_jobs = {
    SHORT: {},
    LONG: {},
    PRIVATE: {}
}

# Maximum parallel jobs per queue type
max_parallel_jobs = {
    SHORT: MAX_PARALLEL_SHORT_JOBS,
    LONG: MAX_PARALLEL_LONG_JOBS,
    PRIVATE: MAX_PARALLEL_PRIVATE_JOBS
}

# Keep track of job results
job_results = {}

# Per-queue locks for thread-safe operations
queue_locks = {
    SHORT: None,
    LONG: None,
    PRIVATE: None
}

# Dictionary to keep track of user's active jobs
user_jobs = {}

# Dictionary to store user rate limiting buckets
user_buckets = {}

# Per-user locks for TOC updates
toc_locks = {}

# LRU cache for persistent TOC data (keyed by refresh_token hash)
TOC_CACHE_MAX_SIZE = int(os.environ.get("TOC_CACHE_MAX_SIZE", "100"))
toc_cache = LRUCache(maxsize=TOC_CACHE_MAX_SIZE)

def get_toc_cache_key(refresh_token: Optional[str]) -> Optional[str]:
    """Generate cache key from refresh_token."""
    if not refresh_token:
        return None
    return hashlib.sha256(refresh_token.encode()).hexdigest()





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

    def eta_to_credits(self, total_seconds):
        """
        Returns 0 if allowance is already available, otherwise seconds until credits are available,
        or infinity if required credits are more than the bucket's max.
        """
        self.update()
        
        # If required credits exceed bucket capacity, return infinity
        if total_seconds > self.max_seconds:
            return float('inf')
        
        # If we already have enough credits, return 0
        if self.seconds_remaining >= total_seconds:
            return 0
        
        # Calculate how many seconds we need to wait for sufficient credits
        needed_seconds = total_seconds - self.seconds_remaining
        wait_time = needed_seconds / self.time_fill_rate
        
        return wait_time

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


async def calculate_queue_time(queue_to_use, running_jobs, exclude_last=False):
    """
    Calculate the estimated time remaining for jobs in queue and running jobs

    Args:
        queue_to_use: Queue to check
        running_jobs: Dictionary of currently running jobs
        exclude_last: Whether to exclude the last job in queue from calculation

    Returns:
        time_ahead_seconds: Estimated time in seconds (float)
    """
    # Skip queue time calculations for private queue
    if queue_to_use == queues[PRIVATE]:
        return 0.0
    
    # Determine which lock to use based on the queue
    queue_type = None
    for qt, q in queues.items():
        if q == queue_to_use:
            queue_type = qt
            break
    
    if queue_type is None:
        return 0.0
    time_ahead = 0
    

    # Add remaining time of currently running jobs
    for running_job_id, running_job in running_jobs.items():
        # Calculate remaining time based on elapsed time since transcription started
        if hasattr(running_job, 'transcribe_start_time') and running_job.transcribe_start_time:
            elapsed_time = time.time() - running_job.transcribe_start_time
            elapsed_time_scaled = elapsed_time / SPEEDUP_FACTOR
            remaining_duration = max(0, running_job.duration - elapsed_time_scaled)
            time_ahead += remaining_duration
        else:
            # If no start time info yet, add full duration
            time_ahead += running_job.duration

    # Add duration of queued jobs
    queue_jobs = list(queue_to_use.queue)
    if exclude_last and queue_jobs:
        queue_jobs = queue_jobs[:-1]  # Exclude the last job

    time_ahead += sum(job.duration for job in queue_jobs)

    # Apply speedup factor
    time_ahead /= SPEEDUP_FACTOR

    # Return time in seconds
    return time_ahead



async def queue_job(job_id, user_email, filename, duration, runpod_token="", language="he", refresh_token: Optional[str] = None):
    # Try to add the job to the queue
    log_message(f"{user_email}: Queuing job {job_id}...")

    # Check if user already has a job queued or running
    if user_email in user_jobs:
        return False, JSONResponse({"error": "יש לך כבר עבודה בתור או בביצוע. אנא המתן לסיומה לפני העלאת קובץ חדש."}, status_code=400)

    # Check rate limits only if not using custom RunPod credentials
    custom_runpod_credentials = bool(runpod_token)
    if not custom_runpod_credentials:
        user_bucket = get_user_quota(user_email)
        eta_seconds = user_bucket.eta_to_credits(duration)
        
        if eta_seconds > 0:
            remaining_minutes = user_bucket.get_remaining_minutes()
            log_message(f"{user_email}: Job queuing rate limited for user {user_email}. Requested: {duration/60:.1f}min, Remaining: {remaining_minutes:.1f}min")
            
            if eta_seconds == float('inf'):
                error_msg = f"הקובץ המבוקש גדול מדי ועובר את מגבלת השימוש החופשי הכוללת. אנא השתמש במפתח פרטי בעזרת ההוראות בסרטון הבא: https://youtu.be/xr8RQRFERLs"
            else:
                wait_minutes = math.ceil(eta_seconds / 60)
                error_msg = f"עברת את מגבלת השימוש החופשי. אנא המתן {wait_minutes} דקות לפני העלאת קובץ חדש, או השתמש במפתח פרטי בעזרת ההוראות בסרטון הבא: https://youtu.be/xr8RQRFERLs"
            
            return False, JSONResponse({"error": error_msg}, status_code=429)

    try:
        job_desc = box.Box()
        job_desc.qtime = time.time()
        job_desc.utime = time.time()
        job_desc.id = job_id
        job_desc.filename = filename
        job_desc.user_email = user_email
        job_desc.duration = duration
        job_desc.runpod_token = runpod_token
        job_desc.uses_custom_runpod = bool(runpod_token)
        job_desc.language = language
        job_desc.refresh_token = refresh_token

        # Determine queue type based on job characteristics
        if job_desc.uses_custom_runpod:
            # Private queue for jobs with custom RunPod credentials
            queue_to_use = queues[PRIVATE]
            job_type = PRIVATE
            running_jobs_to_update = running_jobs[PRIVATE]
        elif duration <= SHORT_JOB_THRESHOLD:
            queue_to_use = queues[SHORT]
            job_type = SHORT
            running_jobs_to_update = running_jobs[SHORT]
        else:
            queue_to_use = queues[LONG]
            job_type = LONG
            running_jobs_to_update = running_jobs[LONG]

        job_desc.job_type = job_type

        # Use the appropriate queue lock
        async with queue_locks[job_type]:
            queue_depth = queue_to_use.qsize()
            queue_to_use.put_nowait(job_desc)

            # Calculate time ahead only for the relevant queue
            time_ahead_seconds = await calculate_queue_time(queue_to_use, running_jobs_to_update, exclude_last=True)
            time_ahead_str = str(timedelta(seconds=int(time_ahead_seconds)))

            # Add job to user's active jobs
            user_jobs[user_email] = job_id

            capture_event(job_id, "job-queued", {"user": user_email, "queue-depth": queue_depth, "job-type": job_type, "custom-runpod": job_desc.uses_custom_runpod})

            log_message(
                f"{user_email}: Job queued successfully: {job_id}, queue depth: {queue_depth}, job type: {job_type}, job desc: {job_desc}"
            )

            return True, (
                JSONResponse({"job_id": job_id, "queued": queue_depth, "job_type": job_type, "time_ahead": time_ahead_str})
            )
    except queue.Full:
        capture_event(job_id, "job-queue-failed", {"queue-depth": queue_depth, "job-type": job_type})

        log_message(f"{user_email}: Job queuing failed: {job_id}")

        cleanup_temp_file(job_id)
        return False, JSONResponse({"error": "השרת עמוס כרגע. אנא נסה שוב מאוחר יותר."}, status_code=503)


@app.get("/")
async def index(request: Request):
    if in_dev:
        user_email = args.dev_user_email or os.environ.get("TS_USER_EMAIL", "dev@example.com")
        session_id = set_user_email(request, user_email)
        response = templates.TemplateResponse("index.html", {"request": request})
        response.set_cookie(key="session_id", value=session_id, httponly=True, secure=not in_dev)
        return response

    user_email = get_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login")

    if in_hiatus_mode:
        return templates.TemplateResponse("server-down.html", {"request": request})
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/languages")
async def list_languages():
    """Return language configuration for client UI."""
    langs = {
        key: {
            "enabled": cfg.get("enabled", False),
            "general_availability": cfg.get("general_availability", False),
        }
        for key, cfg in LANG_CONFIG.get("languages", {}).items()
    }
    return JSONResponse({"languages": langs})


@app.get("/appdata/toc")
async def get_toc(request: Request):
    """Get TOC (table of contents) with all transcription metadata, augmented with in-memory job states."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not refresh_token:
        return JSONResponse({"error": "Not authenticated with Google Drive"}, status_code=401)
    
    if not user_email:
        return JSONResponse({"error": "User email not found"}, status_code=401)
    
    # Load persistent TOC (cached in download_toc, only contains completed jobs)
    toc_data = await download_toc(refresh_token)
    
    if toc_data is None:
        return JSONResponse({"error": "Failed to load TOC"}, status_code=500)
    
    # Make a copy to avoid modifying cached data
    toc_data = copy.deepcopy(toc_data)
    
    # Get TOC version from environment
    toc_version = os.environ.get("TOC_VER", "1.0")
    
    # Augment with in-memory jobs for this user
    in_memory_entries = []
    
    # Check all queues for jobs from this user
    for queue_type in [SHORT, LONG, PRIVATE]:
        queue_to_use = queues[queue_type]
        running_jobs_to_use = running_jobs[queue_type]
        
        # Use lock for thread-safe queue access
        async with queue_locks[queue_type]:
            # Check queued jobs
            for job_desc in list(queues[queue_type].queue):
                if job_desc.user_email == user_email:
                    # Calculate ETA for queued jobs (only for non-private queues)
                    eta_seconds = None
                    if queue_type != PRIVATE:
                        # Find position of this job in queue
                        queue_list = list(queue_to_use.queue)
                        job_position = queue_list.index(job_desc)
                        
                        # Calculate time for jobs ahead of this one
                        jobs_ahead = queue_list[:job_position]
                        time_ahead = sum(job.duration for job in jobs_ahead)
                        
                    # Add remaining time from running jobs
                    for running_job_id, running_job in running_jobs_to_use.items():
                        # Calculate remaining time based on elapsed time since transcription started
                        if hasattr(running_job, 'transcribe_start_time') and running_job.transcribe_start_time:
                            elapsed_time = time.time() - running_job.transcribe_start_time
                            elapsed_time_scaled = elapsed_time / SPEEDUP_FACTOR
                            remaining_duration = max(0, running_job.duration - elapsed_time_scaled)
                            time_ahead += remaining_duration
                        else:
                            time_ahead += running_job.duration
                        
                        # Apply speedup factor
                        eta_seconds = time_ahead / SPEEDUP_FACTOR
                    
                    entry = {
                        "job_id": job_desc.id,
                        "source_filename": job_desc.filename,
                        "language": job_desc.language,
                        "duration_seconds": job_desc.duration,
                        "submitted_at": datetime.fromtimestamp(job_desc.qtime).isoformat(),
                        "status": "Queued",
                        "toc_version": toc_version,
                    }
                    if eta_seconds is not None:
                        entry["eta_seconds"] = int(eta_seconds)
                    in_memory_entries.append(entry)
            
            # Check running jobs
            for job_id, job_desc in running_jobs[queue_type].items():
                if job_desc.user_email == user_email:
                    # Calculate ETA for running jobs (only for non-private queues)
                    eta_seconds = None
                    if queue_type != PRIVATE:
                        # Calculate remaining time for this job based on elapsed time
                        if hasattr(job_desc, 'transcribe_start_time') and job_desc.transcribe_start_time:
                            elapsed_time = time.time() - job_desc.transcribe_start_time
                            elapsed_time_scaled = elapsed_time / SPEEDUP_FACTOR
                            remaining_duration = max(0, job_desc.duration - elapsed_time_scaled)
                        else:
                            remaining_duration = job_desc.duration
                        
                        # Calculate queue time for jobs ahead (queued jobs + other running jobs)
                        # Create a dict of other running jobs (excluding this one)
                        other_running_jobs = {k: v for k, v in running_jobs_to_use.items() if k != job_id}
                        queue_time = await calculate_queue_time(queue_to_use, other_running_jobs, exclude_last=False)
                        
                        # ETA = queue time + remaining time for this job
                        eta_seconds = queue_time + remaining_duration
                    
                    entry = {
                        "job_id": job_desc.id,
                        "source_filename": job_desc.filename,
                        "language": job_desc.language,
                        "duration_seconds": job_desc.duration,
                        "submitted_at": datetime.fromtimestamp(job_desc.qtime).isoformat(),
                        "status": "Being processed",
                        "toc_version": toc_version,
                    }
                    if eta_seconds is not None:
                        entry["eta_seconds"] = int(eta_seconds)
                    in_memory_entries.append(entry)
    
    # Prepend in-memory entries (they should appear first)
    if "entries" not in toc_data:
        toc_data["entries"] = []
    toc_data["entries"] = in_memory_entries + toc_data["entries"]
    
    return JSONResponse(toc_data)


@app.get("/appdata/results/{results_id}")
async def get_transcription_results(results_id: str, request: Request):
    """Download transcription results by UUID (returns gzipped JSON for client-side decompression)."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    
    if not refresh_token:
        return JSONResponse({"error": "Not authenticated with Google Drive"}, status_code=401)
    
    # Find the results file (gzipped)
    filename = f"{results_id}.json.gz"
    file_id = await find_google_appdata_file_by_name(refresh_token, filename)
    
    if not file_id:
        return JSONResponse({"error": "Results file not found"}, status_code=404)
    
    # Download gzipped file as bytes
    file_content = await download_google_appdata_file_bytes(refresh_token, file_id)
    
    if file_content is None:
        return JSONResponse({"error": "Failed to download results"}, status_code=500)
    
    # Return gzipped data for client-side decompression
    from fastapi.responses import Response
    return Response(
        content=file_content,
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "google_analytics_tag": os.environ["GOOGLE_ANALYTICS_TAG"]})

@app.get("/authorize")
async def authorize(request: Request):
    """Redirect to Google OAuth"""
    # Generate state parameter for security
    state = str(uuid.uuid4())
    session_id = get_session_id(request)
    
    if session_id not in sessions:
        sessions[session_id] = {}
    sessions[session_id]["oauth_state"] = state
    
    # Build Google OAuth URL using v2 endpoints
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": GOOGLE_REDIRECT_URI,
        # Request identity scopes + Drive AppData for storing results
        "scope": "openid email profile https://www.googleapis.com/auth/drive.appdata",
        "access_type": "offline",
        "prompt": "consent",
        "state": state
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    
    response = RedirectResponse(url=auth_url)
    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=not in_dev)
    return response

async def get_max_serverless_concurrency(api_key: str) -> Optional[int]:
    """Get maximum serverless concurrency from RunPod"""
    GRAPHQL_URL = "https://api.runpod.io/graphql"
    GRAPHQL_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query {
        myself {
            maxServerlessConcurrency
        }
    }
    """
    
    try:
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                GRAPHQL_URL, 
                headers=GRAPHQL_HEADERS, 
                json={"query": query}
            ) as response:
                response.raise_for_status()
                
                data = await response.json()
                if "errors" in data:
                    logger.error(f"GraphQL Error: {data['errors']}")
                    return None
                    
                max_concurrency = data["data"]["myself"]["maxServerlessConcurrency"]
                log_message(f"Max serverless concurrency: {max_concurrency}")
                return max_concurrency
            
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching max serverless concurrency: {e}")
        return None

async def get_current_worker_usage(api_key: str) -> Optional[int]:
    """Get current worker usage by summing workersMax from all endpoints"""
    GRAPHQL_URL = "https://api.runpod.io/graphql"
    GRAPHQL_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query Endpoints {
        myself {
            endpoints {
                id
                name
                workersMax
                workersMin
                pods {
                    desiredStatus
                }
                scalerType
                scalerValue
                templateId
            }
        }
    }
    """
    
    try:
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                GRAPHQL_URL, 
                headers=GRAPHQL_HEADERS, 
                json={"query": query}
            ) as response:
                response.raise_for_status()
                
                data = await response.json()
                if "errors" in data:
                    logger.error(f"GraphQL Error: {data['errors']}")
                    return None
                    
                endpoints = data["data"]["myself"]["endpoints"]
                total_workers = sum(endpoint.get("workersMax", 0) for endpoint in endpoints)
                log_message(f"Current total worker usage across {len(endpoints)} endpoints: {total_workers}")
                
                # Log individual endpoint details for debugging
                for endpoint in endpoints:
                    log_message(f"Endpoint {endpoint['name']}: workersMax={endpoint.get('workersMax', 0)}")
                
                return total_workers
            
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching current worker usage: {e}")
        return None

async def check_runpod_balance(api_key: str) -> Optional[dict]:
    """Check RunPod balance using GraphQL API"""
    GRAPHQL_URL = "https://api.runpod.io/graphql"
    GRAPHQL_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query {
        myself {
            clientBalance
            currentSpendPerHr
            spendLimit
        }
    }
    """
    
    try:
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                GRAPHQL_URL, 
                headers=GRAPHQL_HEADERS, 
                json={"query": query}
            ) as response:
                response.raise_for_status()
                
                data = await response.json()
                if "errors" in data:
                    logger.error(f"GraphQL Error: {data['errors']}")
                    return None
                    
                balance_info = data["data"]["myself"]
                return {
                    "clientBalance": balance_info.get('clientBalance', 'N/A'),
                    "currentSpendPerHr": balance_info.get('currentSpendPerHr', 'N/A'),
                    "spendLimit": balance_info.get('spendLimit', 'N/A')
                }
            
    except aiohttp.ClientError as e:
        logger.error(f"Error checking RunPod balance: {e}")
        return None

async def find_runpod_endpoint(api_key: str) -> Optional[dict]:
    """Find autogenerated endpoint with full details including template ID using GraphQL API"""
    GRAPHQL_URL = "https://api.runpod.io/graphql"
    GRAPHQL_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query {
        myself {
            endpoints {
                id
                name
                templateId
                gpuIds
                workersMin
                workersMax
                idleTimeout
                scalerType
                scalerValue
            }
        }
    }
    """
    
    try:
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                GRAPHQL_URL, 
                headers=GRAPHQL_HEADERS, 
                json={"query": query}
            ) as response:
                response.raise_for_status()
                
                data = await response.json()
                if "errors" in data:
                    logger.error(f"GraphQL Error: {data['errors']}")
                    return None
                    
                endpoints = data["data"]["myself"]["endpoints"]
                
                # Look for endpoints matching the pattern autogenerated-endpoint-transcribe-ivrit-ai-*
                pattern_regex = r"autogenerated-endpoint-transcribe-ivrit-ai-\d{8}"
                for endpoint in endpoints:
                    log_message(f"Endpoint: {endpoint['name']}")
                    if re.match(pattern_regex, endpoint["name"]):
                        log_message(f"Found endpoint: {endpoint['id']} ({endpoint['name']}) with template {endpoint['templateId']}")
                        return {
                            "id": endpoint["id"], 
                            "name": endpoint["name"],
                            "templateId": endpoint["templateId"],
                            "gpuIds": endpoint["gpuIds"],
                            "workersMin": endpoint["workersMin"],
                            "workersMax": endpoint["workersMax"],
                            "idleTimeout": endpoint["idleTimeout"],
                            "scalerType": endpoint["scalerType"],
                            "scalerValue": endpoint["scalerValue"]
                        }
                
                logger.warning("No autogenerated endpoints found")
                return None
            
    except aiohttp.ClientError as e:
        logger.error(f"Error finding autogenerated endpoint: {e}")
        return None

async def delete_runpod_endpoint(api_key: str, endpoint_id: str) -> bool:
    """Delete an endpoint using REST API"""
    REST_URL = "https://rest.runpod.io/v1"
    REST_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.delete(
                f"{REST_URL}/endpoints/{endpoint_id}",
                headers=REST_HEADERS
            ) as response:
                response.raise_for_status()
                
                log_message(f"Successfully deleted endpoint: {endpoint_id}")
                return True
            
    except aiohttp.ClientError as e:
        logger.error(f"Error deleting endpoint {endpoint_id}: {e}")
        return False

async def create_runpod_endpoint(api_key: str, template_id: str) -> Optional[dict]:
    """Create a new autogenerated endpoint using REST API with dynamic worker limits"""
    REST_URL = "https://rest.runpod.io/v1"
    REST_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Generate date-based name
    current_date = datetime.now().strftime("%Y%m%d")
    endpoint_name = f"autogenerated-endpoint-transcribe-ivrit-ai-{current_date}"
    
    # Get maximum concurrency and current usage
    max_concurrency = await get_max_serverless_concurrency(api_key)
    current_usage = await get_current_worker_usage(api_key)
    
    if max_concurrency is None or current_usage is None:
        logger.warning("Failed to fetch concurrency limits, using default workersMax=2")
        workers_max = 2
    else:
        # Calculate available workers
        available_workers = max_concurrency - current_usage
        
        # Check if no workers are available
        if available_workers <= 0:
            logger.error(f"No workers available: max={max_concurrency}, current={current_usage}, available={available_workers}")
            return None
        
        # Use minimum of available workers and a reasonable default (e.g., 3)
        workers_max = min(available_workers, 3)
        log_message(f"Concurrency calculation: max={max_concurrency}, current={current_usage}, available={available_workers}, setting workersMax={workers_max}")
    
    endpoint_data = {
        "name": endpoint_name,
        "templateId": template_id,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": workers_max,
        "idleTimeout": 5,
        "executionTimeoutMs": EXECUTION_TIMEOUT_MS
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{REST_URL}/endpoints",
                headers=REST_HEADERS,
                json=endpoint_data
            ) as response:
                response.raise_for_status()
                
                result = await response.json()
                endpoint_id = result.get('id')
                
                log_message(f"Created autogenerated endpoint: {endpoint_id} ({endpoint_name})")
                return {
                    "id": endpoint_id,
                    "name": endpoint_name,
                    "status": result.get('status', 'N/A')
                }
            
    except aiohttp.ClientError as e:
        logger.error(f"Error creating autogenerated endpoint: {e}")
        return None

@app.get("/balance")
async def get_balance(request: Request, runpod_token: str = None):
    """Get RunPod balance for the provided credentials"""
    if not runpod_token:
        return JSONResponse({"error": "Missing RunPod token"}, status_code=400)
    
    balance_info = await check_runpod_balance(runpod_token)
    if balance_info is None:
        return JSONResponse({"error": "Failed to fetch balance"}, status_code=500)
    
    return JSONResponse(balance_info)

async def check_runpod_endpoint(runpod_token: str) -> dict:
    """
    Check for autogenerated endpoint, validate template ID, and recreate if needed.
    Returns a dictionary with action, endpoint info, and whether a wait is needed.
    """
    try:
        # Get the required template ID from environment
        required_template_id = os.environ.get("RUNPOD_TEMPLATE_ID")
        if not required_template_id:
            return {"success": False, "error": "RUNPOD_TEMPLATE_ID environment variable not set"}
        
        # Check if endpoint exists with full details
        endpoint_info = await find_runpod_endpoint(runpod_token)
        
        if endpoint_info:
            # Endpoint found, check if template ID matches
            current_template_id = endpoint_info.get("templateId")
            
            if current_template_id == required_template_id:
                # Template ID matches - endpoint is up to date
                return {
                    "success": True,
                    "action": "up_to_date",
                    "endpoint_id": endpoint_info["id"],
                    "endpoint_name": endpoint_info["name"],
                    "template_id": current_template_id,
                    "needs_wait": False
                }
            else:
                # Template ID doesn't match - delete and recreate
                log_message(f"Template ID mismatch: current={current_template_id}, required={required_template_id}")
                
                # Delete the existing endpoint
                delete_success = await delete_runpod_endpoint(runpod_token, endpoint_info["id"])
                if not delete_success:
                    log_message(f"Failed to delete endpoint {endpoint_info['id']}, but proceeding with creation")
                
                # Create new endpoint with correct template
                new_endpoint = await create_runpod_endpoint(runpod_token, required_template_id)
                
                if new_endpoint:
                    return {
                        "success": True,
                        "action": "updated",
                        "endpoint_id": new_endpoint["id"],
                        "endpoint_name": new_endpoint["name"],
                        "old_template_id": current_template_id,
                        "new_template_id": required_template_id,
                        "status": new_endpoint["status"],
                        "needs_wait": True
                    }
                else:
                    return {"success": False, "error": "Failed to create updated endpoint - no workers available or concurrency limit reached"}
        else:
            # No endpoint found - create a new one
            new_endpoint = await create_runpod_endpoint(runpod_token, required_template_id)
            
            if new_endpoint:
                return {
                    "success": True,
                    "action": "created",
                    "endpoint_id": new_endpoint["id"],
                    "endpoint_name": new_endpoint["name"],
                    "template_id": required_template_id,
                    "status": new_endpoint["status"],
                    "needs_wait": True
                }
            else:
                return {"success": False, "error": "Failed to create endpoint - no workers available or concurrency limit reached"}
            
    except Exception as e:
        logger.error(f"Error in check_runpod_endpoint: {e}")
        return {"success": False, "error": "Internal server error"}


@app.post("/check_endpoint")
async def check_endpoint(request: Request):
    """Check for autogenerated endpoint, validate template ID, and recreate if needed"""
    try:
        body = await request.json()
        runpod_token = body.get("runpod_token")
        
        if not runpod_token:
            return JSONResponse({"error": "Missing RunPod token"}, status_code=400)
        
        result = await check_runpod_endpoint(runpod_token)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse({"error": result["error"]}, status_code=500)
            
    except Exception as e:
        logger.error(f"Error in check_endpoint: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.get("/login/authorized")
async def authorized(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle Google OAuth callback"""
    if error:
        error_message = f"Access denied: {error}"
        return templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
    
    if not code or not state:
        error_message = "Missing authorization code or state"
        return templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
    
    # Verify state parameter
    session_id = get_session_id(request)
    if state != sessions.get(session_id, {}).get("oauth_state"):
        error_message = "Invalid state parameter"
        return templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
    
    try:
        # Exchange code for access token using v2 endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                }
            ) as token_response:
                token_data = await token_response.json()
                
                if "error" in token_data:
                    error_message = f"Token exchange failed: {token_data.get('error_description', 'Unknown error')}"
                    return templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
                
                access_token = token_data["access_token"]
                
                # Get user info using v2 endpoint
                async with session.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}
                ) as user_response:
                    user_data = await user_response.json()
                    
                    # Store user email in session
                    user_email = user_data["email"]
                    set_user_email(request, user_email)
                    # Persist refresh token in the session for later Drive uploads
                    existing_refresh = sessions.get(session_id, {}).get("refresh_token")
                    refresh_token = token_data.get("refresh_token", existing_refresh)
                    if session_id not in sessions:
                        sessions[session_id] = {}
                    sessions[session_id]["refresh_token"] = refresh_token
                    
                    # Clean up OAuth state
                    if session_id in sessions:
                        sessions[session_id].pop("oauth_state", None)
                    
                    response = templates.TemplateResponse("close_window.html", {"request": request, "success": True})
                    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=not in_dev)
                    return response
            
    except Exception as e:
        error_message = f"Authentication failed: {str(e)}"
        return templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    runpod_token: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    user_email = get_user_email(request)

    if in_hiatus_mode:
        capture_event(job_id, "file-upload-hiatus-rejected", {"user": user_email})
        return JSONResponse({"error": "השירות כרגע לא פעיל. אנא נסה שוב מאוחר יותר."}, status_code=503)

    capture_event(job_id, "file-upload", {"user": user_email})

    if not file:
        return JSONResponse({"error": "לא נבחר קובץ. אנא בחר קובץ להעלאה."}, status_code=200)

    if file.filename == "":
        return JSONResponse({"error": "שם הקובץ ריק. אנא בחר קובץ תקין."}, status_code=200)

    filename = secure_filename(file.filename)

    # Determine requested language and model from config
    if not language:
        return JSONResponse({"error": "Missing language"}, status_code=400)
    requested_lang = language  # assume already lowercase and correct
    languages_cfg = LANG_CONFIG["languages"]
    if requested_lang not in languages_cfg:
        return JSONResponse({"error": "Unsupported language"}, status_code=400)
    lang_cfg = languages_cfg[requested_lang]

    # Get RunPod token early to determine file size limits
    runpod_token = runpod_token.strip() if runpod_token else ""
    has_private_credentials = bool(runpod_token)

    # Enforce language availability: if not generally available and no private key, reject
    if (not lang_cfg["general_availability"]) and (not has_private_credentials):
        return JSONResponse({"error": "שפה זו זמינה רק לשימוש עם מפתח RunPod פרטי."}, status_code=400)
    
    # Define file size limits
    MAX_FILE_SIZE_REGULAR = 300 * 1024 * 1024  # 300MB
    MAX_FILE_SIZE_PRIVATE = 3 * 1024 * 1024 * 1024  # 3GB
    CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks
    
    max_file_size = MAX_FILE_SIZE_PRIVATE if has_private_credentials else MAX_FILE_SIZE_REGULAR
    max_file_size_text = "3GB" if has_private_credentials else "300MB"

    # Check content-length header before reading file to validate size early
    content_length = None
    if 'content-length' in request.headers:
        try:
            content_length = int(request.headers['content-length'])
        except (ValueError, TypeError):
            pass

    if content_length is not None and content_length > max_file_size:
        return JSONResponse({"error": f"הקובץ גדול מדי. הגודל המקסימלי המותר הוא {max_file_size_text}."}, status_code=400)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    try:
        # Read file content in chunks to avoid memory overload
        total_size = 0
        with open(temp_file_path, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > max_file_size:
                    os.unlink(temp_file_path)
                    return JSONResponse({"error": f"הקובץ גדול מדי. הגודל המקסימלי המותר הוא {max_file_size_text}."}, status_code=400)
                f.write(chunk)
        
        file_size = total_size
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return JSONResponse({"error": f"העלאת הקובץ נכשלה: {str(e)}"}, status_code=200)

    # Get the MIME type of the file
    filetype = magic.Magic(mime=True).from_file(temp_file_path)

    if not is_ffmpeg_supported_mimetype(filetype):
        return JSONResponse({"error": f"סוג הקובץ {filetype} אינו נתמך. אנא העלה קובץ אודיו או וידאו תקין."}, status_code=200)

    # Get the duration of the audio file
    duration = get_media_duration(temp_file_path)

    if duration is None:
        return JSONResponse({"error": "לא ניתן לקרוא את משך הקובץ. אנא ודא שהקובץ תקין ונסה שוב."}, status_code=200)

    # Check if audio duration exceeds maximum allowed duration
    max_duration_seconds = MAX_AUDIO_DURATION_IN_HOURS * 3600
    if duration > max_duration_seconds:
        cleanup_temp_file(job_id)
        return JSONResponse({"error": f"הקובץ ארוך מדי. המשך המקסימלי המותר הוא {MAX_AUDIO_DURATION_IN_HOURS} שעות ({max_duration_seconds/3600:.1f} שעות), אך הקובץ שלך הוא {duration/3600:.1f} שעות."}, status_code=400)

    # Store the temporary file path
    temp_files[job_id] = temp_file_path
    
    # If using custom RunPod credentials, check endpoint status
    if runpod_token:
        endpoint_result = await check_runpod_endpoint(runpod_token)
        if not endpoint_result["success"]:
            cleanup_temp_file(job_id)
            return JSONResponse({"error": f"שגיאה בבדיקת ה-endpoint: {endpoint_result['error']}"}, status_code=400)
        
        if endpoint_result.get("needs_wait", False):
            cleanup_temp_file(job_id)
            action = endpoint_result.get("action", "updated")
            if action == "created":
                message = "נוצר endpoint חדש עבור המפתח שלך. אנא המתן 3 דקות לפני העלאת קבצים."
            else:  # updated
                message = "ה-endpoint שלך עודכן. אנא המתן 3 דקות לפני העלאת קבצים."
            return JSONResponse({"error": message}, status_code=400)

    # Retrieve Google refresh token from session and enqueue job with it
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    # Use the background event loop to call the async function
    queued, res = await queue_job(job_id, user_email, filename, duration, runpod_token, requested_lang, refresh_token)
    if queued:
        job_results[job_id] = {"results": [], "completion_time": None}
    else:
        cleanup_temp_file(job_id)

    return res




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


@app.get("/download/{job_id}")
async def download_file(job_id: str, request: Request):
    if job_id not in temp_files:
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(temp_files[job_id])





async def process_segment(job_id, segment, duration):
    """Process a single segment and update job results"""
    # Convert segment to dict using dataclasses.asdict
    segment_dict = dataclasses.asdict(segment)
    
    # Clean text
    segment_dict['text'] = clean_some_unicode_from_text(segment.text)
    
    # Clean word text
    for word in segment_dict['words']:
        word['word'] = clean_some_unicode_from_text(word['word'])

    job_results[job_id]["results"].append(segment_dict)

    return True


async def transcribe_job(job_desc):
    job_id = job_desc.id
    segs = None

    try:
        log_message(f"{job_desc.user_email}: beginning transcription of {job_desc}, file name={job_desc.filename}")

        temp_file_path = temp_files[job_id]
        duration = job_desc.duration

        # Consume quota only if not using custom RunPod credentials
        if not job_desc.uses_custom_runpod:
            user_bucket = get_user_quota(job_desc.user_email)
            user_bucket.consume(duration)
            remaining_minutes = user_bucket.get_remaining_minutes()
            log_message(f"{job_desc.user_email}: consumed {duration/60:.1f} minutes from quota. Remaining: {remaining_minutes:.1f} minutes")
        else:
            log_message(f"{job_desc.user_email}: using custom RunPod credentials, skipping quota consumption")

        transcribe_start_time = time.time()
        # Store the start time in the job description for ETA calculation
        job_desc.transcribe_start_time = transcribe_start_time
        capture_event(
            job_id,
            "transcribe-start",
            {"user": job_desc.user_email, "queued_seconds": transcribe_start_time - job_desc.qtime, "job-type": job_desc.job_type, "custom-runpod": job_desc.uses_custom_runpod},
        )

        log_message(f"{job_desc.user_email}: job {job_desc} has a duration of {duration} seconds")

        # Load model using ivrit package
        # Pass custom RunPod credentials if provided
        if job_desc.uses_custom_runpod:
            api_key = job_desc.runpod_token
            # Find the autogenerated endpoint
            endpoint_info = await find_runpod_endpoint(api_key)
            if not endpoint_info:
                log_message(f"{job_desc.user_email}: failed to find autogenerated endpoint")
                raise Exception("No autogenerated endpoint found for the provided API key")
            endpoint_id = endpoint_info["id"]
            log_message(f"{job_desc.user_email}: using custom RunPod credentials - found endpoint: {endpoint_id}")
        else:
            api_key = os.environ["RUNPOD_API_KEY"]
            endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
            log_message(f"{job_desc.user_email}: using default RunPod credentials")
        
        # Resolve model by language from config
        lang_cfg = LANG_CONFIG["languages"][job_desc.language]
        selected_model = lang_cfg["model"]
        m = ivrit.load_model(engine='runpod', model=selected_model, api_key=api_key, endpoint_id=endpoint_id, core_engine='stable-whisper')
        
        # Process streaming results
        try:
            if in_dev:
                # In dev mode, use the local file
                segs = m.transcribe_async(path=temp_file_path, diarize=True)
            else:
                # In production mode, send file as URL
                base_url = os.environ["BASE_URL"]
                download_url = urljoin(base_url, f"/download/{job_id}")
                segs = m.transcribe_async(url=download_url, diarize=True)
            
            async for segment in segs:
                await process_segment(job_id, segment, duration)

        except Exception as e:
            log_message(f"Exception during transcription: {e}")
            print(traceback.format_exc())
            raise e
        
        log_message(f"{job_desc.user_email}: done transcribing job {job_id}, audio duration was {duration}.")

        transcribe_done_time = time.time()
        capture_event(
            job_id,
            "transcribe-done",
            {
                "user": job_desc.user_email,
                "transcription_seconds": transcribe_done_time - job_desc.transcribe_start_time,
                "audio_duration_seconds": duration,
                "job-type": job_desc.job_type,
                "language": job_desc.language,
                "custom-runpod": job_desc.uses_custom_runpod,
            },
        )

        job_results[job_id]["completion_time"] = datetime.now()

        # After completion, store results separately and update TOC
        try:
            # Get TOC version from environment
            toc_version = os.environ.get("TOC_VER", "1.0")

            completed_at_iso = job_results[job_id]["completion_time"].isoformat()
            results_id = str(uuid.uuid4())
            
            # Create full payload with both metadata AND results
            # This allows TOC to be rebuilt from individual files if needed
            full_payload = {
                "results_id": results_id,
                "job_id": job_id,
                "source_filename": job_desc.filename,
                "language": job_desc.language,
                "duration_seconds": duration,
                "completed_at": completed_at_iso,
                "toc_version": toc_version,
                "results": job_results[job_id]["results"],
            }
            
            # Upload results file first (before TOC update)
            # This ensures results exist before TOC references them
            results_filename = f"{results_id}.json.gz"
            # Compress JSON data
            json_data = json.dumps(full_payload).encode('utf-8')
            file_data = gzip.compress(json_data)
            mime_type = "application/gzip"
            upload_success = await upload_to_google_appdata(
                job_desc.refresh_token,
                results_filename,
                file_data,
                mime_type,
                job_desc.user_email
            )
            
            if not upload_success:
                logger.error(f"Failed to upload results file for {job_id}, skipping TOC update")
                return
                        
            # Create TOC entry for completed job
            toc_entry = {
                "results_id": results_id,
                "job_id": job_id,
                "source_filename": job_desc.filename,
                "language": job_desc.language,
                "duration_seconds": duration,
                "completed_at": completed_at_iso,
                "status": "Ready",
                "toc_version": toc_version,
            }
            
            # Acquire per-user lock for TOC updates
            toc_lock = get_toc_lock(job_desc.user_email)
            async with toc_lock:
                # Download current TOC
                toc_data = await download_toc(job_desc.refresh_token)
                
                # Append new entry
                if "entries" not in toc_data:
                    toc_data["entries"] = []
                toc_data["entries"].append(toc_entry)
                
                # Upload updated TOC atomically
                await upload_toc(job_desc.refresh_token, toc_data, job_desc.user_email)
            
            log_message(f"{job_desc.user_email}: Uploaded transcription to TOC (results_id: {results_id})")
        except Exception as e:
            logger.error(f"Failed to upload transcription for {job_id}: {e}")
    except Exception as e:
        log_message(f"Error in transcription job {job_id}: {str(e)}")
    finally:
        # Close segs if it exists
        #if segs:
            #try:
            #    segs.close()
            #except Exception as e:
            #    log_message(f"Failed to close runpod job: {e}")
        
        # Remove job from the appropriate running jobs dictionary
        del running_jobs[job_desc.job_type][job_id]
        # Remove job from user's active jobs
        user_email = job_desc.user_email
        if user_email in user_jobs and user_jobs[user_email] == job_id:
            del user_jobs[user_email]
        cleanup_temp_file(job_id)
        # The job thread will terminate itself in the next iteration of the transcribe_job function


async def submit_next_task(job_queue, running_jobs, max_parallel_jobs, queue_type):
    async with queue_locks[queue_type]:
        # Submit all possible tasks until max_parallel_jobs are reached or queue is empty
        while len(running_jobs) < max_parallel_jobs and not job_queue.empty():
            job_desc = job_queue.get()
            running_jobs[job_desc.id] = job_desc
            # Create async task for transcription
            asyncio.create_task(transcribe_job(job_desc))


async def cleanup_old_results():
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




async def event_loop():
    while True:
        await submit_next_task(queues[SHORT], running_jobs[SHORT], max_parallel_jobs[SHORT], SHORT)
        await submit_next_task(queues[LONG], running_jobs[LONG], max_parallel_jobs[LONG], LONG)
        await submit_next_task(queues[PRIVATE], running_jobs[PRIVATE], max_parallel_jobs[PRIVATE], PRIVATE)
        await cleanup_old_results()
        await asyncio.sleep(0.1)


# Global async resources
queue_locks = {
    SHORT: None,
    LONG: None,
    PRIVATE: None
}

if __name__ == "__main__":
    port = 4600 if in_dev else 4500
    uvicorn.run(app, host="0.0.0.0", port=port)
