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

import aiohttp
from fastapi.responses import (
    JSONResponse,
    FileResponse,
    RedirectResponse,
    HTMLResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
import uvicorn
import os
import sys
import math
import time
import json
import uuid
import box
import dotenv
import random
import tempfile
import asyncio
import queue
import asyncio.subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlencode
from functools import wraps
from typing import Optional
from contextlib import asynccontextmanager
import dataclasses
from pathlib import Path

import glob
import traceback
import copy
import hashlib
import gzip
import io

import posthog
import ffmpeg
import imageio_ffmpeg
import base64
import subprocess
import argparse
import re
import ivrit
import json as _json
from cachetools import LRUCache
import yt_dlp

dotenv.load_dotenv()

# PyInstaller support: detect if running from a bundle
def get_base_path():
    """Get the base path for the application, handling PyInstaller bundles."""
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running as a normal Python script
        return Path(__file__).parent.absolute()

BASE_PATH = get_base_path()

# Import file storage backends (will be conditionally imported after args parsing)
from local_file_utils import LocalFileStorageBackend
from file_utils import FileStorageBackend

# Parse CLI arguments for configuration
parser = argparse.ArgumentParser(description='Transcription service with rate limiting')
parser.add_argument('--max-minutes-per-week', type=int, default=180, help='Maximum credit grant in minutes per week (default: 420)')
parser.add_argument('--staging', action='store_true', help='Enable staging mode')
parser.add_argument('--hiatus', action='store_true', help='Enable hiatus mode')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('--dev', action='store_true', help='Enable development mode')
parser.add_argument('--dev-user-email', help='User email for development mode')
parser.add_argument('--dev-https', action='store_true', help='Enable HTTPS in development mode with self-signed certificates')
parser.add_argument('--dev-cert-folder', help='Path to folder containing self-signed certificate files (cert.pem and key.pem)')
parser.add_argument('--local', action='store_true', help='Enable local mode')
parser.add_argument('--data-dir', dest='data_dir', default=None, help='Directory for storing data files (local storage, etc.)')
parser.add_argument('--models-dir', dest='models_dir', default=None, help='Directory containing model files for local mode')
parser.add_argument('--config', dest='config_path', default='config.json', help='Path to configuration JSON defining languages and models (default: config.json)')
parser.add_argument('--max-batch-local', type=int, default=100, help='Max parallel transcription jobs per user in local mode (default: 100)')
parser.add_argument('--max-batch-private', type=int, default=20, help='Max parallel transcription jobs per user with private RunPod key (default: 20)')
args, unknown = parser.parse_known_args()

in_dev = args.staging or args.dev
in_hiatus_mode = args.hiatus or os.environ.get("TS_HIATUS_MODE", "0") == "1"
verbose = args.verbose
in_local_mode = args.local

# Rate limiting configuration from CLI arguments
MAX_MINUTES_PER_WEEK = float('inf') if in_local_mode else args.max_minutes_per_week  # Maximum credit grant per week
REPLENISH_RATE_MINUTES_PER_DAY = MAX_MINUTES_PER_WEEK / 7  # Automatically derive daily replenish rate
MAX_BATCH_LOCAL = args.max_batch_local
MAX_BATCH_PRIVATE = args.max_batch_private

# Import Google Drive backend only if not in local mode (to avoid requiring env vars)
if not in_local_mode:
    from gdrive_file_utils import GoogleDriveStorageBackend

# Initialize file storage backend based on mode
if in_local_mode:
    # Use custom data directory if provided, otherwise use default
    local_data_dir = args.data_dir if args.data_dir else "local_data"
    file_storage_backend: FileStorageBackend = LocalFileStorageBackend(base_dir=local_data_dir)
else:
    file_storage_backend: FileStorageBackend = GoogleDriveStorageBackend()

# Load language/model configuration (mandatory, after args are parsed)
CONFIG_PATH = args.config_path
# Handle PyInstaller: resolve relative paths relative to BASE_PATH
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = str(BASE_PATH / CONFIG_PATH)

APP_CONFIG = {}
with open(CONFIG_PATH, "r", encoding="utf-8") as _cfg_f:
    APP_CONFIG = _json.load(_cfg_f)

# Basic validation of configuration
if "languages" not in APP_CONFIG or not isinstance(APP_CONFIG["languages"], dict):
    raise RuntimeError("Invalid configuration: missing 'languages' dictionary")
for lang_key, lang_cfg in APP_CONFIG["languages"].items():
    if not isinstance(lang_cfg, dict):
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': must be an object")
    
    # Require both ct2_model and ggml_model fields
    if "ct2_model" not in lang_cfg or not isinstance(lang_cfg["ct2_model"], str) or not lang_cfg["ct2_model"].strip():
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'ct2_model'")
    if "ggml_model" not in lang_cfg or not isinstance(lang_cfg["ggml_model"], str) or not lang_cfg["ggml_model"].strip():
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'ggml_model'")
    
    if "general_availability" not in lang_cfg or not isinstance(lang_cfg["general_availability"], bool):
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'general_availability' (bool)")
    if "enabled" not in lang_cfg or not isinstance(lang_cfg["enabled"], bool):
        raise RuntimeError(f"Invalid configuration for language '{lang_key}': missing or invalid 'enabled' (bool)")

# Extract quota increase URL from config (mandatory)
if "quota_increase_url" not in APP_CONFIG:
    raise RuntimeError("Invalid configuration: missing 'quota_increase_url' field")
if not isinstance(APP_CONFIG["quota_increase_url"], str) or not APP_CONFIG["quota_increase_url"].strip():
    raise RuntimeError("Invalid configuration: 'quota_increase_url' must be a non-empty string")
QUOTA_INCREASE_URL = APP_CONFIG["quota_increase_url"]

# Configure logging EARLY - before any other imports that might create loggers
LOG_FORMAT = "[%(asctime)s] %(message)s"
LOGGER_NAME = "transcribe_service"

# Determine log directory (use data directory in local mode, otherwise script directory)
if args.local and args.data_dir:
    # In local mode with data directory specified, write logs there
    log_dir = Path(args.data_dir)
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
else:
    # Otherwise use script directory
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle - use script directory for logs
        log_dir = Path(sys.executable).parent
    else:
        log_dir = Path(__file__).parent.absolute()

LOG_FILE_PATH = str(log_dir / "app.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

# Add file handler for app.log (no console handler - logs only to file)
file_handler = RotatingFileHandler(
    filename=LOG_FILE_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(file_handler)

# Log a test message to verify file handler is working
root_logger.info("File logging initialized - this message should appear in app.log")

logger = logging.getLogger(LOGGER_NAME)
logger.info("Logger configured for transcribe_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global queue_locks, transcoding_lock, backend_version
    
    # Startup
    # Initialize locks
    queue_locks[SHORT] = asyncio.Lock()
    queue_locks[LONG] = asyncio.Lock()
    queue_locks[PRIVATE] = asyncio.Lock()
    transcoding_lock = asyncio.Lock()
    
    # Generate backend version identifier for cache busting
    backend_version = str(random.randint(100000000, 999999999))
    log_message(f"Backend version: {backend_version}")
    
    # Start background event loop
    asyncio.create_task(event_loop())
    
    yield

# Create FastAPI app
app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)

# Mount static files (handle PyInstaller paths)
app.mount("/static", StaticFiles(directory=str(BASE_PATH / "static")), name="static")

# Templates (handle PyInstaller paths)
templates = Jinja2Templates(directory=str(BASE_PATH / "templates"))

# Session management (simplified for FastAPI)
sessions = {}

# Backend version identifier for cache busting (set on startup)
backend_version = None



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


async def require_google_login(request: Request):
    """Ensure the current request belongs to a signed-in Google user."""
    if in_dev or in_local_mode:
        return

    session_id = request.cookies.get("session_id")
    session = sessions.get(session_id) if session_id else None

    if not session or not session.get("user_email"):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )

# Import Google OAuth constants only if not in local mode
# (gdrive_auth will be imported conditionally to avoid requiring env vars in local mode)
if not in_local_mode:
    from gdrive_auth import (
        refresh_google_access_token,
        get_access_token_from_refresh,
        GoogleDriveError,
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        GOOGLE_REDIRECT_URI,
    )
else:
    # Dummy values for local mode (won't be used)
    GOOGLE_CLIENT_ID = None
    GOOGLE_CLIENT_SECRET = None
    GOOGLE_REDIRECT_URI = None
    GoogleDriveError = Exception

# Required OAuth2 scopes for Google authentication
REQUIRED_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.file"
]

def log_message(message):
    logger.info(f"{message}")

def get_user_identifier(request: Request = None, refresh_token: Optional[str] = None, user_email: Optional[str] = None, session_id: Optional[str] = None) -> Optional[str]:
    """Get user identifier for file storage backend.
    
    In local mode: uses user_email or session_id
    In Google Drive mode: uses refresh_token
    """
    if in_local_mode:
        if user_email:
            return user_email
        if session_id:
            return session_id
        if request:
            return get_user_email(request) or get_session_id(request)
        return None
    else:
        return refresh_token

async def download_toc(refresh_token: Optional[str], user_email: Optional[str] = None, session_id: Optional[str] = None) -> dict:
    """Download TOC file (gzipped), using cache if available."""
    user_identifier = get_user_identifier(refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    cache_key = get_toc_cache_key(refresh_token) if not in_local_mode else (user_email or session_id)
    
    # Try to get from cache first (currently disabled)
    if TOC_CACHE_ENABLED and cache_key:
        cached_toc = toc_cache.get(cache_key)
        if cached_toc is not None:
            return copy.deepcopy(cached_toc)
    
    # Not in cache, download from storage
    file_id = await file_storage_backend.find_file_by_name("toc.json.gz", user_identifier)
    
    if not file_id:
        toc_data = {"entries": []}
    else:
        # Download raw bytes and decompress
        file_bytes = await file_storage_backend.download_file_bytes(file_id, user_identifier)
        if not file_bytes:
            toc_data = {"entries": []}
        else:
            file_bytes = gzip.decompress(file_bytes)
            toc_data = json.loads(file_bytes)
    
    # Store in cache
    if TOC_CACHE_ENABLED and cache_key:
        toc_cache[cache_key] = copy.deepcopy(toc_data)
    
    return toc_data

async def upload_toc(refresh_token: Optional[str], toc_data: dict, user_email: Optional[str] = None, session_id: Optional[str] = None) -> bool:
    """Upload TOC file (gzipped), updating existing one atomically or creating new one."""
    user_identifier = get_user_identifier(refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    # Find existing toc.json.gz
    existing_id = await file_storage_backend.find_file_by_name("toc.json.gz", user_identifier)
    
    # Prepare data: compress JSON
    json_data = json.dumps(toc_data).encode('utf-8')
    file_data = gzip.compress(json_data)
    mime_type = "application/gzip"
    
    success = False
    if existing_id:
        # Update existing file atomically
        success = await file_storage_backend.update_file(existing_id, file_data, mime_type, user_identifier, user_email)
    else:
        # Create new file (gzipped)
        success = await file_storage_backend.upload_file("toc.json.gz", file_data, mime_type, user_identifier, user_email)
    
    # Invalidate cache if upload was successful
    if success:
        cache_key = get_toc_cache_key(refresh_token) if not in_local_mode else (user_email or session_id)
        if TOC_CACHE_ENABLED and cache_key:
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
MAX_PARALLEL_PRIVATE_JOBS = 1 if in_local_mode else 1000
MAX_PARALLEL_JOBS_PER_USER = 1
MAX_PARALLEL_TRANSCODES = 4
MAX_QUEUED_JOBS = 20
MAX_QUEUED_PRIVATE_JOBS = 5000
SHORT_JOB_THRESHOLD = 20 * 60

# Set speedup factor based on mode and platform
if in_local_mode and sys.platform == "darwin":
    SPEEDUP_FACTOR = 8  # macOS local mode
else:
    SPEEDUP_FACTOR = 15  # Remote/cloud mode or non-macOS

log_message(f"Using SPEEDUP_FACTOR={SPEEDUP_FACTOR} (local_mode={in_local_mode}, platform={sys.platform})")

TRANSCODING_SPEEDUP = 100  # Transcoding speedup factor for ETA calculation
SUBMISSION_DELAY = 15  # Additional delay in seconds added to ETA calculations
MAX_AUDIO_DURATION_IN_HOURS = 20
EXECUTION_TIMEOUT_MS = int(MAX_AUDIO_DURATION_IN_HOURS * 3600 * 1000 / SPEEDUP_FACTOR)

MAX_FILE_SIZE_REGULAR = 300 * 1024 * 1024  # 300MB
MAX_FILE_SIZE_PRIVATE = 3 * 1024 * 1024 * 1024  # 3GB
UPLOAD_CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks

# Dictionary to store temporary file paths
temp_files = {}

# Transcoding queue and running jobs
transcoding_queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)
transcoding_running_jobs = {}

# Transcoding queue lock
transcoding_lock = None

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

# Upload streaming subscribers per job_id
upload_event_streams = {}


async def emit_upload_event(job_id: str, event_type: str, payload: Optional[dict] = None):
    """Push an event to any streaming client waiting on job_id."""
    queue = upload_event_streams.get(job_id)
    if not queue:
        return

    event = {"type": event_type, "job_id": job_id}
    if payload:
        event.update(payload)
    await queue.put(event)


async def emit_upload_error(
    job_id: str,
    error_key: str,
    *,
    status_code: int = 400,
    i18n_vars: Optional[dict] = None,
    details: Optional[str] = None,
):
    """Send a terminal error event to the streaming client."""
    payload = {
        "error": error_key,
        "i18n_key": error_key,
        "status_code": status_code,
    }
    if i18n_vars:
        payload["i18n_vars"] = i18n_vars
    if details:
        payload["details"] = details
    await emit_upload_event(job_id, "error", payload)


async def upload_event_generator(job_id: str):
    """Async generator that streams events for a specific upload job."""
    queue = upload_event_streams.get(job_id)
    if queue is None:
        return

    try:
        while True:
            event = await queue.get()
            yield (json.dumps(event) + "\n").encode("utf-8")
            if event.get("type") in {"error", "transcoding_complete"}:
                break
    finally:
        upload_event_streams.pop(job_id, None)

# Per-queue locks for thread-safe operations
queue_locks = {
    SHORT: None,
    LONG: None,
    PRIVATE: None
}

# Statistics tracking (global counters since app launch)
stats_lock = asyncio.Lock()
stats_jobs_transcribed = {SHORT: 0, LONG: 0, PRIVATE: 0}
stats_minutes_transcribed = {SHORT: 0.0, LONG: 0.0, PRIVATE: 0.0}
stats_total_jobs_started = 0
stats_total_minutes_processed = 0.0
stats_app_start_time = time.time()

# Heartbeat tracking for local mode auto-shutdown
HEARTBEAT_INTERVAL_SECONDS = 60  # Client sends heartbeat every 60 seconds
HEARTBEAT_MISSED_THRESHOLD = 3  # Number of consecutive missed heartbeats before shutdown
missed_heartbeat_count = 0  # Counter for consecutive missed heartbeats
heartbeat_received_this_period = False  # Flag to track if heartbeat received in current period
last_heartbeat_check_time = None  # Last time we checked for heartbeat

# Google Drive error tracking
stats_gdrive_errors = {
    "toc_upload": 0,
    "toc_download": 0,
    "audio_upload": 0,
    "audio_download": 0,
    "rename": 0,
    "delete": 0
}

# Transcoding statistics
stats_transcoding_jobs = 0
stats_transcoding_total_gb = 0.0
stats_transcoding_total_duration_seconds = 0.0

# Quota statistics
stats_quota_denied = 0

# Dictionary to keep track of user's active jobs
user_jobs = {}

# Dictionary to store user rate limiting buckets
user_buckets = {}

# Per-user locks for TOC updates
toc_locks = {}

# LRU cache for persistent TOC data (keyed by refresh_token hash)
TOC_CACHE_MAX_SIZE = int(os.environ.get("TOC_CACHE_MAX_SIZE", "100"))
toc_cache = LRUCache(maxsize=TOC_CACHE_MAX_SIZE)
TOC_CACHE_ENABLED = False

# LRU cache for Drive file IDs to avoid repeated lookups during streaming
# Key: (refresh_token_hash_prefix, filename) -> file_id
DRIVE_FILE_ID_CACHE_SIZE = 1000
drive_file_id_cache = LRUCache(maxsize=DRIVE_FILE_ID_CACHE_SIZE)

def get_toc_cache_key(refresh_token: Optional[str]) -> Optional[str]:
    """Generate cache key from refresh_token."""
    if not refresh_token:
        return None
    return hashlib.sha256(refresh_token.encode()).hexdigest()


def get_drive_file_id_cache_key(refresh_token: str, filename: str) -> str:
    """Generate cache key for drive file ID lookup."""
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()[:16]
    return f"{token_hash}:{filename}"


async def find_drive_file_by_name_cached(refresh_token: str, filename: str, user_email: Optional[str] = None, session_id: Optional[str] = None) -> Optional[str]:
    """Find file by name with caching to reduce API calls."""
    user_identifier = get_user_identifier(refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    cache_key = get_drive_file_id_cache_key(refresh_token, filename) if not in_local_mode else f"{user_identifier}:{filename}"
    
    # Try cache first
    cached_file_id = drive_file_id_cache.get(cache_key)
    if cached_file_id is not None:
        return cached_file_id
    
    # Cache miss - do the lookup
    file_id = await file_storage_backend.find_file_by_name(filename, user_identifier)
    
    # Cache the result if found
    if file_id:
        drive_file_id_cache[cache_key] = file_id
    
    return file_id





YOUTUBE_URL_PATTERN = re.compile(
    r'^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[A-Za-z0-9_-]{11}'
)

def validate_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.match(url))


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


async def get_media_duration(file_path):
    try:
        # Use ffmpeg directly to get media duration (async to avoid blocking event loop)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_exe, "-i", file_path, "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()
        # ffmpeg outputs info to stderr
        output = stderr_bytes.decode("utf-8", errors="replace")
        # Parse duration from output (format: Duration: HH:MM:SS.ms)
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output)
        if match:
            hours, minutes, seconds = match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            return duration
        return None
    except Exception as e:
        log_message(f"ffmpeg duration error: {str(e)}")
        return None


async def transcode_to_opus(
    input_path: str,
    output_path: str,
    *,
    progress_callback=None,
    duration_hint: Optional[float] = None,
) -> bool:
    """
    Transcode audio file to Opus format with progress reporting.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        '-hide_banner',
        '-loglevel', 'error',
        '-nostats',
        '-progress', 'pipe:1',
        '-i', input_path,
        '-ac', '2',
        '-ar', '48000',
        '-c:a', 'libopus',
        '-b:a', '64k',
        '-vbr', 'off',
        '-application', 'audio',
        '-threads', '1',
        output_path
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def emit_progress(out_time_ms: str):
        if not progress_callback:
            return
        try:
            progress_seconds = int(out_time_ms) / 1_000_000
        except (ValueError, TypeError):
            return
        try:
            await progress_callback(progress_seconds, duration_hint)
        except Exception as exc:
            log_message(f"Progress callback failed: {exc}")

    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode(errors="ignore").strip()
            if not decoded or "=" not in decoded:
                continue
            key, value = decoded.split("=", 1)
            if key == "out_time_ms":
                await emit_progress(value)
            elif key == "progress" and value == "end":
                break
    except Exception as exc:
        log_message(f"Transcoding progress loop failed: {exc}")

    stderr_data = await process.stderr.read()
    return_code = await process.wait()

    if return_code != 0:
        stderr_text = stderr_data.decode(errors="ignore")
        log_message(f"Transcoding error (code {return_code}): {stderr_text}")
        return False

    return True


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
            # Remaining time = duration - (elapsed_time * SPEEDUP_FACTOR)
            remaining_duration = max(0, running_job.duration - elapsed_time * SPEEDUP_FACTOR)
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



async def queue_job(job_id, user_email, filename, duration, runpod_token="", language="he", refresh_token: Optional[str] = None, save_audio: bool = False):
    # Try to add the job to the queue
    log_message(f"{user_email}: Queuing job {job_id}...")
    global stats_quota_denied

    def build_error(error_key, *, status_code=400, i18n_vars=None):
        payload = {"error": error_key, "i18n_key": error_key, "status_code": status_code}
        if i18n_vars:
            payload["i18n_vars"] = i18n_vars
        return False, payload

    # Check if user has reached their batch limit
    if bool(runpod_token):
        user_batch_limit = MAX_BATCH_PRIVATE
    elif in_local_mode:
        user_batch_limit = MAX_BATCH_LOCAL
    else:
        user_batch_limit = 1

    active_count = len(user_jobs.get(user_email, set()))
    if active_count >= user_batch_limit:
        return build_error("errorBatchLimitReached", status_code=400)

    max_duration_seconds = MAX_AUDIO_DURATION_IN_HOURS * 3600
    if duration > max_duration_seconds:
        return build_error(
            "errorFileTooLong",
            status_code=400,
            i18n_vars={
                "maxHours": MAX_AUDIO_DURATION_IN_HOURS,
                "maxSeconds": f"{max_duration_seconds/3600:.1f}",
                "fileHours": f"{duration/3600:.1f}",
            },
        )

    # Check rate limits only if not using custom RunPod credentials
    custom_runpod_credentials = bool(runpod_token)
    if not custom_runpod_credentials:
        user_bucket = get_user_quota(user_email)
        eta_seconds = user_bucket.eta_to_credits(duration)
        
        if eta_seconds > 0:
            remaining_minutes = user_bucket.get_remaining_minutes()
            log_message(f"{user_email}: Job queuing rate limited for user {user_email}. Requested: {duration/60:.1f}min, Remaining: {remaining_minutes:.1f}min")
            
            if eta_seconds == float('inf'):
                async with stats_lock:
                    stats_quota_denied += 1
                return build_error("errorFileTooLargeForFreeService", status_code=429)
            else:
                wait_minutes = math.ceil(eta_seconds / 60)
                async with stats_lock:
                    stats_quota_denied += 1
                return build_error(
                    "errorRateLimitExceeded",
                    status_code=429,
                    i18n_vars={"minutes": wait_minutes},
                )

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
        job_desc.save_audio = save_audio

        # Determine queue type based on job characteristics
        if job_desc.uses_custom_runpod or in_local_mode:
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
            user_jobs.setdefault(user_email, set()).add(job_id)

            capture_event(job_id, "job-queued", {"user": user_email, "queue-depth": queue_depth, "job-type": job_type, "custom-runpod": job_desc.uses_custom_runpod})

            log_message(
                f"{user_email}: Job queued successfully: {job_id}, queue depth: {queue_depth}, job type: {job_type}, job desc: {job_desc}"
            )

            return True, {
                "job_id": job_id,
                "queue_depth": queue_depth,
                "job_type": job_type,
                "time_ahead_display": time_ahead_str,
                "time_ahead_seconds": int(time_ahead_seconds),
            }
    except queue.Full:
        capture_event(job_id, "job-queue-failed", {"queue-depth": queue_depth, "job-type": job_type})

        log_message(f"{user_email}: Job queuing failed: {job_id}")

        cleanup_temp_file(job_id)
        return build_error("errorServerBusy", status_code=503)


@app.get("/", dependencies=[Depends(require_google_login)])
async def index(request: Request):
    if in_dev or in_local_mode:
        user_email = args.dev_user_email or os.environ.get("TS_USER_EMAIL", "local@example.com")
        session_id = set_user_email(request, user_email)
        response = templates.TemplateResponse("index.html", {
            "request": request,
            "quota_increase_url": QUOTA_INCREASE_URL,
            "in_local_mode": in_local_mode
        })
        response.set_cookie(key="session_id", value=session_id, httponly=True, secure=not (in_dev or in_local_mode))
        response.headers["ETag"] = f'"{backend_version}"'
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    user_email = get_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login")

    if in_hiatus_mode:
        response = templates.TemplateResponse("server-down.html", {"request": request})
        response.headers["ETag"] = f'"{backend_version}"'
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response
    
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "quota_increase_url": QUOTA_INCREASE_URL,
        "in_local_mode": in_local_mode
    })
    response.headers["ETag"] = f'"{backend_version}"'
    response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/languages", dependencies=[Depends(require_google_login)])
async def list_languages():
    """Return language configuration for client UI."""
    langs = {
        key: {
            "enabled": cfg.get("enabled", False),
            "general_availability": cfg.get("general_availability", False),
        }
        for key, cfg in APP_CONFIG.get("languages", {}).items()
    }
    return JSONResponse({
        "languages": langs,
        "batch": {
            "max_batch_local": MAX_BATCH_LOCAL,
            "max_batch_private": MAX_BATCH_PRIVATE,
            "max_batch_default": 1,
        },
    })


@app.get("/appdata/toc", dependencies=[Depends(require_google_login)])
async def get_toc(request: Request):
    """Get TOC (table of contents) with all transcription metadata, augmented with in-memory job states."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    if not user_email:
        return JSONResponse({"error": "errorUserEmailNotFound", "i18n_key": "errorUserEmailNotFound"}, status_code=401)
    
    # Load persistent TOC (cached in download_toc, only contains completed jobs)
    toc_data = await download_toc(refresh_token, user_email=user_email, session_id=session_id)
    
    if toc_data is None:
        return JSONResponse({"error": "errorTocLoadFailed", "i18n_key": "errorTocLoadFailed"}, status_code=500)
    
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
                                # Remaining time = duration - (elapsed_time * SPEEDUP_FACTOR)
                                remaining_duration = max(0, running_job.duration - elapsed_time * SPEEDUP_FACTOR)
                                time_ahead += remaining_duration
                            else:
                                time_ahead += running_job.duration
                        
                        # Apply speedup factor to total time ahead
                        eta_seconds = time_ahead / SPEEDUP_FACTOR
                        # Add submission delay
                        eta_seconds += SUBMISSION_DELAY
                    
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
                            # Remaining time = duration - (elapsed_time * SPEEDUP_FACTOR)
                            remaining_duration = max(0, job_desc.duration - elapsed_time * SPEEDUP_FACTOR)
                        else:
                            remaining_duration = job_desc.duration
                        
                        # ETA is based on the remaining time for this job plus submission delay.
                        # Other jobs in the queue do not delay the completion of this already running task.
                        eta_seconds = (remaining_duration / SPEEDUP_FACTOR) + SUBMISSION_DELAY
                    
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


@app.get("/appdata/results/{results_id}", dependencies=[Depends(require_google_login)])
async def get_transcription_results(results_id: str, request: Request):
    """Download transcription results by UUID (returns gzipped JSON for client-side decompression)."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    # Find the results file (gzipped)
    filename = f"{results_id}.json.gz"
    file_id = await file_storage_backend.find_file_by_name(filename, user_identifier)
    
    if not file_id:
        return JSONResponse({"error": "errorResultsNotFound", "i18n_key": "errorResultsNotFound"}, status_code=404)
    
    # Download gzipped file as bytes
    file_content = await file_storage_backend.download_file_bytes(file_id, user_identifier)
    
    if file_content is None:
        return JSONResponse({"error": "errorResultsDownloadFailed", "i18n_key": "errorResultsDownloadFailed"}, status_code=500)
    
    # Return gzipped data for client-side decompression
    from fastapi.responses import Response
    return Response(
        content=file_content,
        media_type="application/gzip",
        headers={
            "Cache-Control": "max-age=864000",
        },
    )


@app.get("/appdata/edits/{results_id}", dependencies=[Depends(require_google_login)])
async def get_edits(results_id: str, request: Request):
    """Get edit data for a transcription by UUID."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    # Find the edits file
    filename = f"{results_id}.edits.json.gz"
    file_id = await file_storage_backend.find_file_by_name(filename, user_identifier)
    
    if not file_id:
        return JSONResponse({"error": "errorEditsNotFound", "i18n_key": "errorEditsNotFound"}, status_code=404)
    
    # Download file as bytes
    file_content = await file_storage_backend.download_file_bytes(file_id, user_identifier)
    
    if file_content is None:
        return JSONResponse({"error": "errorEditsDownloadFailed", "i18n_key": "errorEditsDownloadFailed"}, status_code=500)
    
    # Decompress the gzipped data
    try:
        decompressed_data = gzip.decompress(file_content)
    except Exception as e:
        logger.error(f"Failed to decompress edits file: {e}")
        return JSONResponse({"error": "errorEditsDecompressionFailed", "i18n_key": "errorEditsDecompressionFailed"}, status_code=500)
    
    # Return JSON data
    from fastapi.responses import Response
    return Response(
        content=decompressed_data,
        media_type="application/json",
        headers={
            "Cache-Control": "no-cache",
        },
    )


@app.post("/appdata/edits/{results_id}", dependencies=[Depends(require_google_login)])
async def save_edits(results_id: str, request: Request):
    """Save edit data for a transcription by UUID."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    # Get the edits data from request body
    try:
        body = await request.json()
        edits = body.get("edits", {})
        speaker_names = body.get("speakerNames", {})
        speaker_swaps = body.get("speakerSwaps", {})
    except Exception as e:
        logger.error(f"Failed to parse edits data: {e}")
        return JSONResponse({"error": "errorInvalidData", "i18n_key": "errorInvalidData"}, status_code=400)
    
    # Serialize edits to JSON and compress with gzip
    import json
    edits_json = json.dumps({
        "edits": edits, 
        "speakerNames": speaker_names,
        "speakerSwaps": speaker_swaps
    }, ensure_ascii=False)
    json_data = edits_json.encode('utf-8')
    file_data = gzip.compress(json_data)
    mime_type = "application/gzip"
    
    # Find existing edits file
    filename = f"{results_id}.edits.json.gz"
    existing_id = await file_storage_backend.find_file_by_name(filename, user_identifier)
    
    success = False
    if existing_id:
        # Update existing file
        success = await file_storage_backend.update_file(existing_id, file_data, mime_type, user_identifier, user_email)
    else:
        # Upload new file
        file_id = await file_storage_backend.upload_file(filename, file_data, mime_type, user_identifier, user_email)
        success = file_id is not None
    
    if success:
        return JSONResponse({"success": True})
    else:
        return JSONResponse({"error": "errorEditsSaveFailed", "i18n_key": "errorEditsSaveFailed"}, status_code=500)


@app.get("/appdata/audio/{results_id}", dependencies=[Depends(require_google_login)])
async def get_audio_file(results_id: str, request: Request):
    """Download opus audio file by results_id UUID."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    # Find the opus file
    filename = f"{results_id}.opus"
    file_id = await file_storage_backend.find_file_by_name(filename, user_identifier)
    
    if not file_id:
        return JSONResponse({"error": "errorAudioNotFound", "i18n_key": "errorAudioNotFound"}, status_code=404)
    
    # Download opus file as bytes
    file_content = await file_storage_backend.download_file_bytes(file_id, user_identifier)
    
    if file_content is None:
        async with stats_lock:
            stats_gdrive_errors["audio_download"] += 1
        return JSONResponse({"error": "errorAudioDownloadFailed", "i18n_key": "errorAudioDownloadFailed"}, status_code=500)
    
    # Return opus audio file
    from fastapi.responses import Response
    return Response(
        content=file_content,
        media_type="audio/ogg",
        headers={
            "Cache-Control": "max-age=864000",
        },
    )


@app.api_route("/appdata/audio/stream/{results_id}", methods=["GET", "HEAD"], dependencies=[Depends(require_google_login)])
async def stream_audio_file(results_id: str, request: Request):
    """Stream opus audio file with range support for seeking."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    # Find the opus file (using cache to avoid repeated lookups for range requests)
    filename = f"{results_id}.opus"
    file_id = await find_drive_file_by_name_cached(refresh_token, filename, user_email=user_email, session_id=session_id)
    
    if not file_id:
        return JSONResponse({"error": "errorAudioNotFound", "i18n_key": "errorAudioNotFound"}, status_code=404)
    
    # For HEAD requests, verify file exists and return headers without content
    if request.method == "HEAD":
        metadata = await file_storage_backend.get_file_metadata(file_id, user_identifier)
        
        from fastapi.responses import Response
        return Response(
            status_code=200,
            media_type="audio/ogg",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "max-age=864000",
                "Content-Length": str(metadata.get("size", 0) if metadata else 0),
            },
        )
    
    # Get range header from request
    range_header = request.headers.get("range")
    
    # Stream file with range support
    result = await file_storage_backend.stream_file_range(file_id, range_header, user_identifier)
    
    if result is None:
        async with stats_lock:
            stats_gdrive_errors["audio_download"] += 1
        return JSONResponse({"error": "errorAudioStreamFailed", "i18n_key": "errorAudioStreamFailed"}, status_code=500)
    
    content, status_code, start_byte, end_byte, total_size = result
    
    # Build response headers
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "max-age=864000",
    }
    
    if status_code == 206:
        # Partial content response
        headers["Content-Range"] = f"bytes {start_byte}-{end_byte}/{total_size}"
        headers["Content-Length"] = str(len(content))
    
    from fastapi.responses import Response
    return Response(
        content=content,
        status_code=status_code,
        media_type="audio/ogg",
        headers=headers,
    )


@app.post("/appdata/rename", dependencies=[Depends(require_google_login)])
async def rename_file(request: Request):
    """Rename a file in the TOC by updating its source_filename."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    if not user_email:
        return JSONResponse({"error": "errorUserEmailNotFound", "i18n_key": "errorUserEmailNotFound"}, status_code=401)
    
    try:
        body = await request.json()
        results_id = body.get("results_id")
        new_filename = body.get("new_filename")
        
        if not results_id or not new_filename:
            return JSONResponse({"error": "errorMissingNewFilename", "i18n_key": "errorMissingNewFilename"}, status_code=400)
        
        # Sanitize new filename
        new_filename = new_filename.strip()
        if not new_filename:
            return JSONResponse({"error": "errorFilenameEmpty", "i18n_key": "errorFilenameEmpty"}, status_code=400)
        
        # Acquire per-user lock for TOC updates
        toc_lock = get_toc_lock(user_email)
        async with toc_lock:
            # Download current TOC
            toc_data = await download_toc(refresh_token, user_email=user_email, session_id=session_id)

            if not toc_data or "entries" not in toc_data:
                async with stats_lock:
                    stats_gdrive_errors["toc_download"] += 1
                return JSONResponse({"error": "errorTocLoadFailed", "i18n_key": "errorTocLoadFailed"}, status_code=500)

            # Find the entry with matching results_id
            entry_found = False
            for entry in toc_data["entries"]:
                if entry.get("results_id") == results_id:
                    entry["source_filename"] = new_filename
                    entry_found = True
                    break

            if not entry_found:
                return JSONResponse({"error": "errorFileNotInToc", "i18n_key": "errorFileNotInToc"}, status_code=404)

            # Upload updated TOC atomically
            success = await upload_toc(refresh_token, toc_data, user_email=user_email, session_id=session_id)

            if not success:
                async with stats_lock:
                    stats_gdrive_errors["toc_upload"] += 1
                return JSONResponse({"error": "errorTocUpdateFailed", "i18n_key": "errorTocUpdateFailed"}, status_code=500)
        
        logging.info(f"{user_email}: Renamed file {results_id} to {new_filename}")
        return JSONResponse({"success": True, "new_filename": new_filename})
        
    except Exception as e:
        logging.error(f"Error renaming file: {e}")
        async with stats_lock:
            stats_gdrive_errors["rename"] += 1
        return JSONResponse({"error": "errorInternalServer", "i18n_key": "errorInternalServer"}, status_code=500)


@app.post("/appdata/delete", dependencies=[Depends(require_google_login)])
async def delete_file(request: Request):
    """Delete a file by removing it from TOC and deleting associated files (opus, json.gz)."""
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    if not user_email:
        return JSONResponse({"error": "errorUserEmailNotFound", "i18n_key": "errorUserEmailNotFound"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    try:
        body = await request.json()
        results_id = body.get("results_id")
        
        if not results_id:
            return JSONResponse({"error": "errorMissingResultsId", "i18n_key": "errorMissingResultsId"}, status_code=400)
        
        # Acquire per-user lock for TOC updates
        toc_lock = get_toc_lock(user_email)
        async with toc_lock:
            # Download current TOC
            toc_data = await download_toc(refresh_token, user_email=user_email, session_id=session_id)

            if not toc_data or "entries" not in toc_data:
                async with stats_lock:
                    stats_gdrive_errors["toc_download"] += 1
                return JSONResponse({"error": "errorTocLoadFailed", "i18n_key": "errorTocLoadFailed"}, status_code=500)

            # Find and remove the entry with matching results_id
            entry_found = False
            original_entries_count = len(toc_data["entries"])
            toc_data["entries"] = [
                entry for entry in toc_data["entries"]
                if entry.get("results_id") != results_id
            ]

            if len(toc_data["entries"]) < original_entries_count:
                entry_found = True

            if not entry_found:
                return JSONResponse({"error": "errorFileNotInToc", "i18n_key": "errorFileNotInToc"}, status_code=404)

            # Upload updated TOC atomically (removing from TOC first)
            success = await upload_toc(refresh_token, toc_data, user_email=user_email, session_id=session_id)

            if not success:
                async with stats_lock:
                    stats_gdrive_errors["toc_upload"] += 1
                return JSONResponse({"error": "errorTocUpdateFailed", "i18n_key": "errorTocUpdateFailed"}, status_code=500)
        
        # After TOC update, delete associated files
        # Delete opus file if it exists
        opus_filename = f"{results_id}.opus"
        opus_file_id = await file_storage_backend.find_file_by_name(opus_filename, user_identifier)
        if opus_file_id:
            delete_success = await file_storage_backend.delete_file(opus_file_id, user_identifier, user_email)
            if not delete_success:
                async with stats_lock:
                    stats_gdrive_errors["delete"] += 1
            # Invalidate file ID cache for deleted file
            cache_key = get_drive_file_id_cache_key(refresh_token, opus_filename) if not in_local_mode else f"{user_identifier}:{opus_filename}"
            drive_file_id_cache.pop(cache_key, None)

        # Delete json.gz results file
        json_filename = f"{results_id}.json.gz"
        json_file_id = await file_storage_backend.find_file_by_name(json_filename, user_identifier)
        if json_file_id:
            delete_success = await file_storage_backend.delete_file(json_file_id, user_identifier, user_email)
            if not delete_success:
                async with stats_lock:
                    stats_gdrive_errors["delete"] += 1
        
        # Delete edits file if it exists
        edits_filename = f"{results_id}.edits.json.gz"
        edits_file_id = await file_storage_backend.find_file_by_name(edits_filename, user_identifier)
        if edits_file_id:
            delete_success = await file_storage_backend.delete_file(edits_file_id, user_identifier, user_email)
            if not delete_success:
                async with stats_lock:
                    stats_gdrive_errors["delete"] += 1
        
        logging.info(f"{user_email}: Deleted file {results_id} from TOC and associated files")
        return JSONResponse({"success": True})
        
    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        async with stats_lock:
            stats_gdrive_errors["delete"] += 1
        return JSONResponse({"error": "errorInternalServer", "i18n_key": "errorInternalServer"}, status_code=500)


@app.get("/appdata/data-dir", dependencies=[Depends(require_google_login)])
async def get_data_dir(request: Request):
    """Return the resolved path to the user's local data directory. Only available in local mode."""
    if not in_local_mode:
        return JSONResponse({"error": "Only available in local mode"}, status_code=403)

    user_email = get_user_email(request)
    if not user_email:
        return JSONResponse({"error": "errorUserEmailNotFound", "i18n_key": "errorUserEmailNotFound"}, status_code=401)

    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)

    user_dir_str = await file_storage_backend.ensure_folder(user_identifier)
    user_dir = Path(user_dir_str).resolve()

    path = str(user_dir)
    # On WSL, convert to a Windows path so users can paste it into Explorer
    if sys.platform == "linux":
        try:
            path = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
        except FileNotFoundError:
            pass  # Not WSL, keep the Linux path

    return JSONResponse({"path": path})


@app.post("/appdata/donate_data", dependencies=[Depends(require_google_login)])
async def donate_data(request: Request):
    """Donate transcription data (audio + transcript + edits) for the ivrit.ai v2 dataset."""
    import tarfile
    
    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")
    user_email = get_user_email(request)
    
    if not in_local_mode and not refresh_token:
        return JSONResponse({"error": "errorNotAuthenticated", "i18n_key": "errorNotAuthenticated"}, status_code=401)
    
    if not user_email:
        return JSONResponse({"error": "errorUserEmailNotFound", "i18n_key": "errorUserEmailNotFound"}, status_code=401)
    
    user_identifier = get_user_identifier(request=request, refresh_token=refresh_token, user_email=user_email, session_id=session_id)
    
    try:
        body = await request.json()
        results_id = body.get("results_id")
        checkbox_confirmed = body.get("checkbox_confirmed", False)
        
        if not results_id:
            return JSONResponse({"error": "errorMissingResultsId", "i18n_key": "errorMissingResultsId"}, status_code=400)
        
        if not checkbox_confirmed:
            return JSONResponse({"error": "errorDonateCheckboxRequired", "i18n_key": "errorDonateCheckboxRequired"}, status_code=400)
        
        # Acquire per-user lock for TOC updates
        toc_lock = get_toc_lock(user_email)
        async with toc_lock:
            # Download current TOC
            toc_data = await download_toc(refresh_token, user_email=user_email, session_id=session_id)

            if not toc_data or "entries" not in toc_data:
                return JSONResponse({"error": "errorTocLoadFailed", "i18n_key": "errorTocLoadFailed"}, status_code=500)

            # Find the entry with matching results_id
            entry = None
            entry_index = None
            for i, e in enumerate(toc_data["entries"]):
                if e.get("results_id") == results_id:
                    entry = e
                    entry_index = i
                    break
            
            if entry is None:
                return JSONResponse({"error": "errorFileNotInToc", "i18n_key": "errorFileNotInToc"}, status_code=404)
            
            # Check if already donated
            if entry.get("donated"):
                return JSONResponse({"error": "errorAlreadyDonated", "i18n_key": "errorAlreadyDonated"}, status_code=400)
            
            # Download opus audio file
            opus_filename = f"{results_id}.opus"
            opus_file_id = await file_storage_backend.find_file_by_name(opus_filename, user_identifier)
            opus_content = None
            if opus_file_id:
                opus_content = await file_storage_backend.download_file_bytes(opus_file_id, user_identifier)
            
            if not opus_content:
                return JSONResponse({"error": "errorDonateNoAudio", "i18n_key": "errorDonateNoAudio"}, status_code=404)
            
            # Download transcript json.gz
            json_filename = f"{results_id}.json.gz"
            json_file_id = await file_storage_backend.find_file_by_name(json_filename, user_identifier)
            transcript_content = None
            if json_file_id:
                transcript_content = await file_storage_backend.download_file_bytes(json_file_id, user_identifier)
            
            if not transcript_content:
                return JSONResponse({"error": "errorDonateNoTranscript", "i18n_key": "errorDonateNoTranscript"}, status_code=404)
            
            # Download edits if they exist (optional)
            edits_filename = f"{results_id}.edits.json.gz"
            edits_file_id = await file_storage_backend.find_file_by_name(edits_filename, user_identifier)
            edits_content = None
            if edits_file_id:
                edits_content = await file_storage_backend.download_file_bytes(edits_file_id, user_identifier)
            
            # Create uploads directory if it doesn't exist
            uploads_dir = Path(args.data_dir or ".") / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            # Create desc.json with metadata
            submission_time = datetime.utcnow().isoformat() + "Z"
            desc_data = {
                "user_email": user_email,
                "submitted_at": submission_time,
                "checkbox_confirmed": checkbox_confirmed,
                "original_filename": entry.get("source_filename", "unknown"),
                "results_id": results_id,
                "language": entry.get("language", "he"),
            }
            desc_json = json.dumps(desc_data, ensure_ascii=False, indent=2).encode('utf-8')
            desc_filename = f"{results_id}.desc.json"
            
            # Create tar file (in memory for local mode, on disk otherwise)
            tar_filename = f"{results_id}.tar"
            tar_buffer = io.BytesIO()
            
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                # Add opus file
                opus_info = tarfile.TarInfo(name=opus_filename)
                opus_info.size = len(opus_content)
                tar.addfile(opus_info, io.BytesIO(opus_content))
                
                # Add transcript (gzipped json)
                json_info = tarfile.TarInfo(name=json_filename)
                json_info.size = len(transcript_content)
                tar.addfile(json_info, io.BytesIO(transcript_content))
                
                # Add edits if they exist
                if edits_content:
                    edits_info = tarfile.TarInfo(name=edits_filename)
                    edits_info.size = len(edits_content)
                    tar.addfile(edits_info, io.BytesIO(edits_content))
                
                # Add desc.json inside the tar
                desc_info = tarfile.TarInfo(name=desc_filename)
                desc_info.size = len(desc_json)
                tar.addfile(desc_info, io.BytesIO(desc_json))
            
            tar_content = tar_buffer.getvalue()
            
            # Update TOC to mark as donated
            toc_data["entries"][entry_index]["donated"] = True
            
            # Upload updated TOC
            success = await upload_toc(refresh_token, toc_data, user_email=user_email, session_id=session_id)
            
            if not success:
                return JSONResponse({"error": "errorTocUpdateFailed", "i18n_key": "errorTocUpdateFailed"}, status_code=500)
            
            # In local mode, return the tar file as a download
            if in_local_mode:
                logging.info(f"{user_email}: Donated data for results_id {results_id} (download)")
                return Response(
                    content=tar_content,
                    media_type="application/x-tar",
                    headers={
                        "Content-Disposition": f'attachment; filename="{tar_filename}"',
                        "X-Donation-Success": "true",
                    },
                )
            
            # In non-local mode, save to uploads directory
            uploads_dir = Path(args.data_dir or ".") / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            tar_path = uploads_dir / tar_filename
            with open(tar_path, "wb") as f:
                f.write(tar_content)
            
            # Write desc.json separately (not in tar, as specified for non-local mode)
            desc_path = uploads_dir / desc_filename
            with open(desc_path, "wb") as f:
                f.write(desc_json)
        
        logging.info(f"{user_email}: Donated data for results_id {results_id}")
        return JSONResponse({"success": True})
        
    except Exception as e:
        logging.error(f"Error donating data: {e}")
        traceback.print_exc()
        return JSONResponse({"error": "errorInternalServer", "i18n_key": "errorInternalServer"}, status_code=500)


@app.get("/login")
async def login(request: Request):
    google_analytics_tag = os.environ.get("GOOGLE_ANALYTICS_TAG", "")
    response = templates.TemplateResponse("login.html", {
        "request": request,
        "google_analytics_tag": google_analytics_tag
    })
    response.headers["ETag"] = f'"{backend_version}"'
    response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

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
        "scope": " ".join(REQUIRED_OAUTH_SCOPES),
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
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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
        
        # Use minimum of available workers and a reasonable default (e.g., 5).
        # Going with 5 (previously had 3 here) as sometimes workers are unavailable, causing long wait times.
        # This somewhat improves time-to-start.
        workers_max = min(available_workers, 5)
        log_message(f"Concurrency calculation: max={max_concurrency}, current={current_usage}, available={available_workers}, setting workersMax={workers_max}")
    
    endpoint_data = {
        "name": endpoint_name,
        "templateId": template_id,
        "gpuTypeIds": [
            # 24GB VRAM options (best for performance)
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX A5000",
            "NVIDIA L4",
            "NVIDIA A30",
            # 48GB VRAM options (highest performance)
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            # 16GB VRAM options (cost-effective)
            "NVIDIA RTX A4000",
            "NVIDIA RTX 2000 Ada Generation"
        ],
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

@app.get("/balance", dependencies=[Depends(require_google_login)])
async def get_balance(request: Request, runpod_token: str = None):
    """Get RunPod balance for the provided credentials"""
    # Block RunPod balance checks in local mode
    if in_local_mode:
        raise HTTPException(status_code=400, detail="RunPod balance check not available in local mode")
    
    if not runpod_token:
        return JSONResponse({"error": "errorMissingRunpodToken", "i18n_key": "errorMissingRunpodToken"}, status_code=400)
    
    balance_info = await check_runpod_balance(runpod_token)
    if balance_info is None:
        return JSONResponse({"error": "errorBalanceFetchFailed", "i18n_key": "errorBalanceFetchFailed"}, status_code=500)
    
    return JSONResponse(balance_info)

@app.get("/quota", dependencies=[Depends(require_google_login)])
async def get_quota(request: Request):
    """Get user's remaining quota"""
    user_email = get_user_email(request)
    
    if not user_email:
        return JSONResponse({"error": "errorUserNotFound", "i18n_key": "errorUserNotFound"}, status_code=400)
    
    user_bucket = get_user_quota(user_email)
    remaining_minutes = user_bucket.get_remaining_minutes()
    
    return JSONResponse({
        "remainingMinutes": remaining_minutes,
        "maxMinutesPerWeek": MAX_MINUTES_PER_WEEK
    })


@app.post("/client_heartbeat")
async def client_heartbeat():
    """Receive heartbeat from client to prevent auto-shutdown in local mode."""
    global missed_heartbeat_count, heartbeat_received_this_period
    
    if not in_local_mode:
        return JSONResponse({"error": "Heartbeat only available in local mode"}, status_code=400)
    
    # Reset missed count and mark heartbeat as received
    missed_heartbeat_count = 0
    heartbeat_received_this_period = True
    
    return JSONResponse({"status": "ok"})

@app.get("/stats", dependencies=[Depends(require_google_login)])
async def get_stats(request: Request):
    """Get application statistics"""
    async with stats_lock:
        # Calculate current queued jobs by type
        queued_jobs = {SHORT: [], LONG: [], PRIVATE: []}
        for queue_type, queue_obj in queues.items():
            for job_desc in list(queue_obj.queue):
                queued_jobs[queue_type].append({
                    "duration": job_desc.duration,
                    "language": job_desc.language,
                    "filename": job_desc.filename
                })

        # Calculate current running jobs by type
        running_jobs_stats = {SHORT: [], LONG: [], PRIVATE: []}
        for queue_type, running_dict in running_jobs.items():
            for job_id, job_desc in running_dict.items():
                running_jobs_stats[queue_type].append({
                    "duration": job_desc.duration,
                    "language": job_desc.language,
                    "filename": job_desc.filename,
                    "elapsed_time": time.time() - getattr(job_desc, 'transcribe_start_time', job_desc.utime)
                })

        # Calculate totals
        def format_duration(minutes):
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours:02d}:{mins:02d}"

        # Calculate uptime
        uptime_seconds = time.time() - stats_app_start_time
        uptime_hours = int(uptime_seconds // 3600)
        uptime_days = uptime_hours // 24
        uptime_display = f"{uptime_days}d {uptime_hours % 24}h" if uptime_days > 0 else f"{uptime_hours}h"

        drive_errors = dict(stats_gdrive_errors)
        transcoding_stats = {
            "jobs": stats_transcoding_jobs,
            "total_gb": stats_transcoding_total_gb,
            "total_duration_seconds": stats_transcoding_total_duration_seconds,
            "total_duration_formatted": format_duration(stats_transcoding_total_duration_seconds / 60.0 if stats_transcoding_total_duration_seconds else 0)
        }

        stats_data = {
            "uptime": uptime_display,
            "uptime_seconds": uptime_seconds,
            "queued_jobs": {
                "short": {
                    "count": len(queued_jobs[SHORT]),
                    "total_duration_minutes": sum(job["duration"] for job in queued_jobs[SHORT]) / 60.0
                },
                "long": {
                    "count": len(queued_jobs[LONG]),
                    "total_duration_minutes": sum(job["duration"] for job in queued_jobs[LONG]) / 60.0
                },
                "private": {
                    "count": len(queued_jobs[PRIVATE]),
                    "total_duration_minutes": sum(job["duration"] for job in queued_jobs[PRIVATE]) / 60.0
                }
            },
            "running_jobs": {
                "short": {
                    "count": len(running_jobs_stats[SHORT]),
                    "total_duration_minutes": sum(job["duration"] for job in running_jobs_stats[SHORT]) / 60.0
                },
                "long": {
                    "count": len(running_jobs_stats[LONG]),
                    "total_duration_minutes": sum(job["duration"] for job in running_jobs_stats[LONG]) / 60.0
                },
                "private": {
                    "count": len(running_jobs_stats[PRIVATE]),
                    "total_duration_minutes": sum(job["duration"] for job in running_jobs_stats[PRIVATE]) / 60.0
                }
            },
            "transcribed_since_launch": {
                "short": {
                    "jobs_count": stats_jobs_transcribed[SHORT],
                    "total_minutes": stats_minutes_transcribed[SHORT],
                    "total_minutes_formatted": format_duration(stats_minutes_transcribed[SHORT])
                },
                "long": {
                    "jobs_count": stats_jobs_transcribed[LONG],
                    "total_minutes": stats_minutes_transcribed[LONG],
                    "total_minutes_formatted": format_duration(stats_minutes_transcribed[LONG])
                },
                "private": {
                    "jobs_count": stats_jobs_transcribed[PRIVATE],
                    "total_minutes": stats_minutes_transcribed[PRIVATE],
                    "total_minutes_formatted": format_duration(stats_minutes_transcribed[PRIVATE])
                },
                "total": {
                    "jobs_count": stats_total_jobs_started,
                    "total_minutes": stats_total_minutes_processed,
                    "total_minutes_formatted": format_duration(stats_total_minutes_processed)
                }
            },
            "system_info": {
                "max_parallel_jobs": {
                    "short": max_parallel_jobs[SHORT],
                    "long": max_parallel_jobs[LONG],
                    "private": max_parallel_jobs[PRIVATE]
                },
            },
            "transcoding": transcoding_stats,
            "errors": {
                "google_drive": drive_errors,
                "quota_denied": stats_quota_denied
            }
        }

        return JSONResponse(stats_data)

async def check_runpod_endpoint(runpod_token: str) -> dict:
    """
    Check for autogenerated endpoint, validate template ID, and recreate if needed.
    Returns a dictionary with action, endpoint info, and whether a wait is needed.
    """
    if in_local_mode:
        raise RuntimeError("RunPod endpoint functions are not available in local mode")
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


@app.post("/check_endpoint", dependencies=[Depends(require_google_login)])
async def check_endpoint(request: Request):
    """Check for autogenerated endpoint, validate template ID, and recreate if needed"""
    # Block RunPod endpoint checks in local mode
    if in_local_mode:
        raise HTTPException(status_code=400, detail="RunPod endpoint check not available in local mode")
    
    try:
        body = await request.json()
        runpod_token = body.get("runpod_token")
        
        if not runpod_token:
            return JSONResponse({"error": "errorMissingRunpodToken", "i18n_key": "errorMissingRunpodToken"}, status_code=400)
        
        result = await check_runpod_endpoint(runpod_token)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse({"error": "errorInternalServer", "i18n_key": "errorInternalServer", "details": result.get("error", "")}, status_code=500)
            
    except Exception as e:
        logger.error(f"Error in check_endpoint: {e}")
        return JSONResponse({"error": "errorInternalServer", "i18n_key": "errorInternalServer"}, status_code=500)

@app.get("/login/authorized")
async def authorized(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle Google OAuth callback"""
    if in_local_mode:
        raise HTTPException(status_code=400, detail="OAuth not available in local mode")
    
    if error:
        error_message = f"Access denied: {error}"
        response = templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
        response.headers["ETag"] = f'"{backend_version}"'
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response
    
    if not code or not state:
        error_message = "Missing authorization code or state"
        response = templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
        response.headers["ETag"] = f'"{backend_version}"'
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response
    
    # Verify state parameter
    session_id = get_session_id(request)
    if state != sessions.get(session_id, {}).get("oauth_state"):
        error_message = "Invalid state parameter"
        response = templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
        response.headers["ETag"] = f'"{backend_version}"'
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response
    
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
                    response = templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
                    response.headers["ETag"] = f'"{backend_version}"'
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
                    return response
                
                # Validate that all required scopes were granted
                required_scopes = set(REQUIRED_OAUTH_SCOPES)
                granted_scopes_str = token_data.get("scope", "")
                granted_scopes = set(granted_scopes_str.split())
                
                # Check if all required scopes are present in granted scopes
                missing_scopes = required_scopes - granted_scopes
                if missing_scopes:
                    logger.warning(f"User did not grant all required scopes. Missing: {missing_scopes}")
                    error_message = "errorDrivePermissionsRequired"
                    # Clean up OAuth state
                    if session_id in sessions:
                        sessions[session_id].pop("oauth_state", None)
                    response = templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message, "i18n_key": True})
                    response.headers["ETag"] = f'"{backend_version}"'
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
                    return response
                
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
                    response.headers["ETag"] = f'"{backend_version}"'
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
                    return response
            
    except Exception as e:
        error_message = f"Authentication failed: {str(e)}"
        response = templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
        response.headers["ETag"] = f'"{backend_version}"'
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


async def queue_transcoding_job(
    *,
    job_id: str,
    filename: str,
    user_email: str,
    input_path: str,
    file_size_bytes: int,
    estimated_duration: float,
    runpod_token: str,
    language: str,
    refresh_token: Optional[str],
    save_audio: bool,
) -> Optional[int]:
    """
    Create a transcoding job descriptor, queue it, initialize results,
    emit a transcoding_waiting event, and kick the transcoding scheduler.

    Returns the queue depth on success, or None if the queue is full.
    """
    transcoding_job = box.Box()
    transcoding_job.id = job_id
    transcoding_job.filename = filename
    transcoding_job.user_email = user_email
    transcoding_job.duration = estimated_duration
    transcoding_job.runpod_token = runpod_token
    transcoding_job.language = language
    transcoding_job.refresh_token = refresh_token
    transcoding_job.input_path = input_path
    transcoding_job.file_size_bytes = file_size_bytes
    transcoding_job.qtime = time.time()
    transcoding_job.save_audio = save_audio

    temp_files[job_id] = input_path

    try:
        async with transcoding_lock:
            queue_depth = transcoding_queue.qsize()
            transcoding_queue.put_nowait(transcoding_job)
    except queue.Full:
        cleanup_temp_file(job_id)
        return None

    job_results[job_id] = {"results": [], "completion_time": None}

    await emit_upload_event(
        job_id,
        "transcoding_waiting",
        {"job_id": job_id, "queue_depth": queue_depth, "filename": filename},
    )

    asyncio.create_task(submit_next_transcoding_task())

    return queue_depth


@app.post("/upload", dependencies=[Depends(require_google_login)])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    runpod_token: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    save_audio: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    user_email = get_user_email(request)
    
    log_message(f"{user_email}: Upload request started - job_id={job_id}, filename={file.filename if file else 'None'}, language={language}, save_audio={save_audio}")

    if in_hiatus_mode:
        log_message(f"{user_email}: Upload rejected - service in hiatus mode (job_id={job_id})")
        capture_event(job_id, "file-upload-hiatus-rejected", {"user": user_email})
        return JSONResponse({"error": "errorServiceUnavailable", "i18n_key": "errorServiceUnavailable"}, status_code=503)

    capture_event(job_id, "file-upload", {"user": user_email})

    if not file:
        log_message(f"{user_email}: No file provided (job_id={job_id})")
        return JSONResponse({"error": "errorNoFileSelected", "i18n_key": "errorNoFileSelected"}, status_code=200)

    if file.filename == "":
        log_message(f"{user_email}: Empty filename (job_id={job_id})")
        return JSONResponse({"error": "errorEmptyFilename", "i18n_key": "errorEmptyFilename"}, status_code=200)

    # Use original filename directly - it's only used for display in TOC, never for filesystem operations
    # Actual files use system-generated temp names and UUID-based names on Google Drive
    filename = file.filename

    content_length = None
    if "content-length" in request.headers:
        try:
            content_length = int(request.headers["content-length"])
        except (ValueError, TypeError):
            pass

    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")

    log_message(f"{user_email}: Validating upload metadata (job_id={job_id}, content_length={content_length})")
    metadata, error_response = await validate_upload_request_metadata(
        request,
        user_email=user_email,
        language=language,
        runpod_token=runpod_token,
        save_audio=save_audio,
        file_size=content_length,
        refresh_token=refresh_token,
    )
    if error_response:
        log_message(f"{user_email}: Upload validation failed (job_id={job_id})")
        return error_response
    
    log_message(f"{user_email}: Upload metadata validated successfully (job_id={job_id})")

    lang_cfg = metadata["lang_cfg"]
    runpod_token = metadata["normalized_runpod_token"]
    has_private_credentials = metadata["has_private_credentials"]
    max_file_size = metadata["max_file_size"]
    max_file_size_text = metadata["max_file_size_text"]
    save_audio_bool = metadata["save_audio_bool"]

    requested_lang = language  # safe after validation

    log_message(f"{user_email}: Creating temp file for upload (job_id={job_id})")
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    log_message(f"{user_email}: Temp file created (job_id={job_id}, path={temp_file_path})")

    try:
        # Read file content in chunks to avoid memory overload
        log_message(f"{user_email}: Starting file read (job_id={job_id})")
        total_size = 0
        with open(temp_file_path, 'wb') as f:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                total_size += len(chunk)
                if max_file_size is not None and total_size > max_file_size:
                    log_message(f"{user_email}: File too large (job_id={job_id}, size={total_size}, max={max_file_size})")
                    os.unlink(temp_file_path)
                    return JSONResponse({
                        "error": "errorFileTooLarge",
                        "i18n_key": "errorFileTooLarge",
                        "i18n_vars": {"size": max_file_size_text}
                    }, status_code=400)
                f.write(chunk)
        
        file_size = total_size
        metadata["file_size_bytes"] = total_size
        log_message(f"{user_email}: File read completed (job_id={job_id}, size={file_size} bytes)")
    except Exception as e:
        log_message(f"{user_email}: ERROR - File read failed (job_id={job_id}): {type(e).__name__}: {str(e)}")
        logger.exception(f"{user_email}: Full traceback for file read failure (job_id={job_id})")
        # Clean up temp file if it exists
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return JSONResponse({
            "error": "errorUploadFailed",
            "i18n_key": "errorUploadFailed",
            "i18n_vars": {"details": str(e)}
        }, status_code=200)

    # Estimate duration based on file size: 1 minute per 1MB (for transcoding progress only)
    estimated_duration = (file_size / (1024 * 1024)) * 60

    event_queue = asyncio.Queue()
    upload_event_streams[job_id] = event_queue

    log_message(f"{user_email}: Queueing transcoding job (job_id={job_id})")
    queue_depth = await queue_transcoding_job(
        job_id=job_id,
        filename=filename,
        user_email=user_email,
        input_path=temp_file_path,
        file_size_bytes=metadata.get("file_size_bytes", file_size),
        estimated_duration=estimated_duration,
        runpod_token=runpod_token,
        language=requested_lang,
        refresh_token=refresh_token,
        save_audio=save_audio_bool,
    )
    if queue_depth is None:
        log_message(f"{user_email}: ERROR - Transcoding queue full (job_id={job_id})")
        return JSONResponse({"error": "errorServerBusy", "i18n_key": "errorServerBusy"}, status_code=503)

    log_message(f"{user_email}: Returning streaming response (job_id={job_id})")
    return StreamingResponse(
        upload_event_generator(job_id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


async def get_youtube_video_info(url: str) -> Optional[dict]:
    """
    Fetch metadata for a YouTube video without downloading it.
    Returns the yt-dlp info dict (with 'duration', 'title', etc.) on success, None on error.
    """
    ydl_opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    def do_extract():
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as e:
            log_message(f"yt-dlp info extraction failed for {url}: {type(e).__name__}: {e}")
            return None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, do_extract)


async def download_youtube_audio(url: str, output_path: str, job_id: str) -> Optional[dict]:
    """
    Download audio from a YouTube URL using yt-dlp.
    Returns the yt-dlp info dict on success, None on failure.
    Emits youtube_download_progress events via upload_event_streams.
    """
    loop = asyncio.get_event_loop()

    def progress_hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                percent = round((downloaded / total) * 100)
            else:
                percent = 0
            try:
                asyncio.run_coroutine_threadsafe(
                    emit_upload_event(job_id, "youtube_download_progress", {"progress_percent": percent}),
                    loop,
                )
            except Exception:
                pass

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": output_path + ".%(ext)s",
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    def do_download():
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info
        except Exception as e:
            log_message(f"yt-dlp download failed for job {job_id}: {type(e).__name__}: {e}")
            return None

    info = await loop.run_in_executor(None, do_download)
    return info


@app.post("/upload/youtube", dependencies=[Depends(require_google_login)])
async def upload_youtube(request: Request):
    job_id = str(uuid.uuid4())
    user_email = get_user_email(request)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "errorInvalidJson", "i18n_key": "errorInvalidJson"}, status_code=400)

    youtube_url = body.get("youtube_url", "").strip()
    language = body.get("language")
    runpod_token = body.get("runpod_token")
    save_audio = body.get("save_audio", "false")
    rights_confirmed = body.get("rights_confirmed", False)

    log_message(f"{user_email}: YouTube upload request started - job_id={job_id}, url={youtube_url}, language={language}")

    if in_hiatus_mode:
        return JSONResponse({"error": "errorServiceUnavailable", "i18n_key": "errorServiceUnavailable"}, status_code=503)

    if not rights_confirmed:
        return JSONResponse({"error": "errorYoutubeRightsNotConfirmed", "i18n_key": "errorYoutubeRightsNotConfirmed"}, status_code=400)

    if not youtube_url or not validate_youtube_url(youtube_url):
        return JSONResponse({"error": "errorInvalidYoutubeUrl", "i18n_key": "errorInvalidYoutubeUrl"}, status_code=400)

    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")

    metadata, error_response = await validate_upload_request_metadata(
        request,
        user_email=user_email,
        language=language,
        runpod_token=runpod_token,
        save_audio=save_audio,
        file_size=None,
        refresh_token=refresh_token,
    )
    if error_response:
        return error_response

    runpod_token = metadata["normalized_runpod_token"]
    max_file_size = metadata["max_file_size"]
    max_file_size_text = metadata["max_file_size_text"]
    save_audio_bool = metadata["save_audio_bool"]
    requested_lang = language

    capture_event(job_id, "youtube-upload", {"user": user_email, "url": youtube_url})

    # --- Pre-download checks: fetch video info without downloading ---
    info = await get_youtube_video_info(youtube_url)
    if info is None:
        log_message(f"{user_email}: YouTube info extraction failed (job_id={job_id})")
        return JSONResponse({"error": "errorYoutubeDownloadFailed", "i18n_key": "errorYoutubeDownloadFailed"}, status_code=400)

    video_duration = info.get("duration") or 0

    # Check duration limit (mirrors queue_job check)
    max_duration_seconds = MAX_AUDIO_DURATION_IN_HOURS * 3600
    if video_duration > max_duration_seconds:
        log_message(f"{user_email}: YouTube video too long ({video_duration}s) (job_id={job_id})")
        return JSONResponse({
            "error": "errorFileTooLong",
            "i18n_key": "errorFileTooLong",
            "i18n_vars": {
                "maxHours": MAX_AUDIO_DURATION_IN_HOURS,
                "maxSeconds": f"{max_duration_seconds/3600:.1f}",
                "fileHours": f"{video_duration/3600:.1f}",
            },
        }, status_code=400)

    # Check credits before downloading (only when not using custom RunPod token)
    custom_runpod_credentials = bool(runpod_token)
    if not custom_runpod_credentials and video_duration > 0:
        user_bucket = get_user_quota(user_email)
        eta_seconds = user_bucket.eta_to_credits(video_duration)

        if eta_seconds > 0:
            if eta_seconds == float('inf'):
                log_message(f"{user_email}: YouTube video too large for free service ({video_duration}s) (job_id={job_id})")
                return JSONResponse({
                    "error": "errorFileTooLargeForFreeService",
                    "i18n_key": "errorFileTooLargeForFreeService",
                }, status_code=429)
            else:
                wait_minutes = math.ceil(eta_seconds / 60)
                log_message(f"{user_email}: YouTube rate limited, wait {wait_minutes}min (job_id={job_id})")
                return JSONResponse({
                    "error": "errorRateLimitExceeded",
                    "i18n_key": "errorRateLimitExceeded",
                    "i18n_vars": {"minutes": wait_minutes},
                }, status_code=429)

    event_queue = asyncio.Queue()
    upload_event_streams[job_id] = event_queue

    async def youtube_download_and_queue():
        temp_file_path = None
        try:
            await emit_upload_event(job_id, "youtube_download_started", {"url": youtube_url})

            # Generate a unique path prefix without creating a file — yt-dlp
            # appends the real extension via %(ext)s, so a pre-existing empty
            # placeholder would confuse the file-resolution logic.
            temp_file_path = os.path.join(tempfile.gettempdir(), f"{job_id}.ytdl")

            dl_info = await download_youtube_audio(youtube_url, temp_file_path, job_id)
            if dl_info is None:
                await emit_upload_error(job_id, "errorYoutubeDownloadFailed")
                return

            # yt-dlp may write to the exact outtmpl path, or append a real
            # extension (e.g. .ytdl.webm).  Use glob to find whatever it created.
            candidates = sorted(
                [p for p in glob.glob(temp_file_path + "*") if os.path.getsize(p) > 0],
                key=os.path.getsize,
                reverse=True,
            )
            if not candidates:
                log_message(f"{user_email}: YouTube download produced no file (job_id={job_id})")
                await emit_upload_error(job_id, "errorYoutubeDownloadFailed")
                return
            actual_path = candidates[0]

            file_size = os.path.getsize(actual_path)
            if max_file_size is not None and file_size > max_file_size:
                await emit_upload_error(
                    job_id, "errorFileTooLarge",
                    i18n_vars={"size": max_file_size_text},
                )
                os.unlink(actual_path)
                return

            # Use title from pre-download info fetch
            video_title = info.get("title", "youtube_audio")
            filename = video_title + ".opus"

            await emit_upload_event(job_id, "youtube_download_complete", {"filename": filename})

            # Use pre-fetched duration if available, fall back to file-size estimate
            estimated_duration = video_duration if video_duration > 0 else (file_size / (1024 * 1024)) * 60

            queue_depth = await queue_transcoding_job(
                job_id=job_id,
                filename=filename,
                user_email=user_email,
                input_path=actual_path,
                file_size_bytes=file_size,
                estimated_duration=estimated_duration,
                runpod_token=runpod_token,
                language=requested_lang,
                refresh_token=refresh_token,
                save_audio=save_audio_bool,
            )
            if queue_depth is None:
                await emit_upload_error(job_id, "errorServerBusy", status_code=503)

        except Exception as e:
            log_message(f"{user_email}: YouTube upload error (job_id={job_id}): {type(e).__name__}: {e}")
            logger.exception(f"{user_email}: YouTube upload traceback (job_id={job_id})")
            # Clean up any files yt-dlp may have created
            if temp_file_path:
                for p in glob.glob(temp_file_path + "*"):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
            await emit_upload_error(job_id, "errorYoutubeDownloadFailed")

    asyncio.create_task(youtube_download_and_queue())

    return StreamingResponse(
        upload_event_generator(job_id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


async def handle_transcoding(job_id: str):
    """
    Handle transcoding of an uploaded file to Opus format.
    After transcoding completes, queues the job for transcription.
    """
    global stats_transcoding_jobs, stats_transcoding_total_gb, stats_transcoding_total_duration_seconds
    if job_id not in transcoding_running_jobs:
        log_message(f"Transcoding job {job_id} not found in running jobs")
        return
    transcoding_job = transcoding_running_jobs[job_id]
    input_path = transcoding_job.input_path
    
    # Create output path for transcoded file
    output_path = input_path + ".opus"

    duration_hint_seconds = None
    try:
        if transcoding_job.duration:
            duration_hint_seconds = float(transcoding_job.duration)
    except Exception:
        duration_hint_seconds = None

    await emit_upload_event(
        job_id,
        "transcoding_started",
        {
            "filename": transcoding_job.filename,
            "file_size_bytes": transcoding_job.get("file_size_bytes"),
        },
    )

    async def progress_callback(progress_seconds: float, duration_hint_value: Optional[float] = None):
        duration_for_percent = duration_hint_value or duration_hint_seconds or transcoding_job.duration or 0
        percent = None
        if duration_for_percent and duration_for_percent > 0:
            percent = max(0.0, min(100.0, (progress_seconds / duration_for_percent) * 100.0))
        payload = {
            "progress_seconds": progress_seconds,
            "duration_seconds": duration_for_percent,
            "progress_percent": percent,
            "filename": transcoding_job.filename,
        }
        await emit_upload_event(job_id, "transcoding_progress", payload)
    
    try:
        success = await transcode_to_opus(
            input_path,
            output_path,
            progress_callback=progress_callback,
            duration_hint=duration_hint_seconds,
        )
        
        if not success:
            log_message(f"Transcoding failed for job {job_id}")
            await emit_upload_error(job_id, "errorTranscodingFailed")
            # Clean up
            if job_id in transcoding_running_jobs:
                del transcoding_running_jobs[job_id]
            cleanup_temp_file(job_id)
            if os.path.exists(output_path):
                os.unlink(output_path)
            return
        
        # Get duration from the generated Opus file before replacing the original
        duration = await get_media_duration(output_path)

        # Replace original file with transcoded file
        try:
            if os.path.exists(input_path):
                os.unlink(input_path)
            os.rename(output_path, input_path)
            temp_files[job_id] = input_path
        except Exception as e:
            log_message(f"Error replacing file after transcoding for job {job_id}: {str(e)}")
            await emit_upload_error(job_id, "errorInternalServer", details="transcoding_replace_failed")
            if job_id in transcoding_running_jobs:
                del transcoding_running_jobs[job_id]
            cleanup_temp_file(job_id)
            if os.path.exists(output_path):
                os.unlink(output_path)
            return
        
        # Update the job with actual duration
        transcoding_job.duration = duration
        
        # Remove from running jobs
        if job_id in transcoding_running_jobs:
            del transcoding_running_jobs[job_id]
        
        # Queue the job for transcription
        queued, queue_info = await queue_job(
            job_id,
            transcoding_job.user_email,
            transcoding_job.filename,
            transcoding_job.duration,
            transcoding_job.runpod_token,
            transcoding_job.language,
            transcoding_job.refresh_token,
            transcoding_job.save_audio
        )
        
        if not queued:
            log_message(f"Failed to queue job {job_id} after transcoding")
            if isinstance(queue_info, dict):
                await emit_upload_event(job_id, "error", queue_info)
            cleanup_temp_file(job_id)
        else:
            await emit_upload_event(
                job_id,
                "queue_position",
                {
                    "queue_depth": queue_info.get("queue_depth"),
                    "job_type": queue_info.get("job_type"),
                    "filename": transcoding_job.filename,
                },
            )
            await emit_upload_event(
                job_id,
                "eta",
                {
                    "eta_seconds": queue_info.get("time_ahead_seconds"),
                    "eta_display": queue_info.get("time_ahead_display"),
                    "job_type": queue_info.get("job_type"),
                },
            )
            await emit_upload_event(
                job_id,
                "transcoding_complete",
                {
                    "queue_depth": queue_info.get("queue_depth"),
                    "job_type": queue_info.get("job_type"),
                },
            )
            async with stats_lock:
                stats_transcoding_jobs += 1
                file_size_gb = 0.0
                if getattr(transcoding_job, "file_size_bytes", None):
                    file_size_gb = transcoding_job.file_size_bytes / (1024 ** 3)
                stats_transcoding_total_gb += file_size_gb
                stats_transcoding_total_duration_seconds += duration
        
    except Exception as e:
        log_message(f"Error in transcoding task for job {job_id}: {str(e)}")
        # Clean up
        if job_id in transcoding_running_jobs:
            del transcoding_running_jobs[job_id]
        cleanup_temp_file(job_id)
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass


def cleanup_temp_file(job_id):
    if job_id in temp_files:
        temp_file_path = temp_files[job_id]
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            log_message(f"Error deleting temporary file: {str(e)}")
        finally:
            del temp_files[job_id]


async def validate_upload_request_metadata(
    request: Request,
    *,
    user_email: Optional[str],
    language: Optional[str],
    runpod_token: Optional[str],
    save_audio: Optional[str],
    file_size: Optional[int],
    refresh_token: Optional[str],
) -> tuple[Optional[dict], Optional[JSONResponse]]:
    if in_hiatus_mode:
        return None, JSONResponse({"error": "errorServiceUnavailable", "i18n_key": "errorServiceUnavailable"}, status_code=503)

    if not user_email:
        return None, JSONResponse({"error": "errorUserEmailNotFound", "i18n_key": "errorUserEmailNotFound"}, status_code=401)

    # Determine batch limit for this user
    if bool(runpod_token):
        user_batch_limit = MAX_BATCH_PRIVATE
    elif in_local_mode:
        user_batch_limit = MAX_BATCH_LOCAL
    else:
        user_batch_limit = 1

    # Count active transcription jobs
    active_job_count = len(user_jobs.get(user_email, set()))

    # Count active transcoding jobs (queued + running)
    transcoding_count = 0
    async with transcoding_lock:
        for queued_job in list(transcoding_queue.queue):
            if queued_job.user_email == user_email:
                transcoding_count += 1
        for existing_job_id, existing_job in transcoding_running_jobs.items():
            if existing_job.user_email == user_email:
                transcoding_count += 1

    total_active = active_job_count + transcoding_count
    if total_active >= user_batch_limit:
        return None, JSONResponse({"error": "errorBatchLimitReached", "i18n_key": "errorBatchLimitReached"}, status_code=400)

    if save_audio is None:
        return None, JSONResponse({"error": "errorMissingSaveAudio", "i18n_key": "errorMissingSaveAudio"}, status_code=400)

    if not in_local_mode and not refresh_token:
        return None, JSONResponse({"error": "errorDriveNotConnected", "i18n_key": "errorDriveNotConnected"}, status_code=401)

    try:
        user_identifier = get_user_identifier(refresh_token=refresh_token, user_email=user_email)
        folder_id = await file_storage_backend.ensure_folder(user_identifier)
    except GoogleDriveError as exc:
        error_text = str(exc).lower()
        if "access_token_scope_insufficient" in error_text or "insufficientpermission" in error_text:
            return None, JSONResponse({"error": "errorDrivePermissionsInsufficient", "i18n_key": "errorDrivePermissionsInsufficient"}, status_code=403)
        logger.error("Drive folder validation failed for %s: %s", user_email, exc)
        return None, JSONResponse({"error": "errorDriveAccessFailed", "i18n_key": "errorDriveAccessFailed"}, status_code=500)

    if not folder_id:
        logger.warning("Drive folder not available during upload validation for %s", user_email)
        return None, JSONResponse({"error": "errorDriveUnavailable", "i18n_key": "errorDriveUnavailable"}, status_code=401)

    normalized_runpod_token = runpod_token.strip() if runpod_token else ""
    has_private_credentials = bool(normalized_runpod_token)

    if not language:
        return None, JSONResponse({"error": "errorMissingLanguage", "i18n_key": "errorMissingLanguage"}, status_code=400)

    languages_cfg = APP_CONFIG["languages"]
    if language not in languages_cfg:
        return None, JSONResponse({"error": "errorUnsupportedLanguage", "i18n_key": "errorUnsupportedLanguage"}, status_code=400)

    lang_cfg = languages_cfg[language]

    if (not lang_cfg["general_availability"]) and (not has_private_credentials):
        return None, JSONResponse({"error": "errorLanguageRequiresPrivateKey", "i18n_key": "errorLanguageRequiresPrivateKey"}, status_code=400)

    # In local mode, no file size limits
    if in_local_mode:
        max_file_size = None
        max_file_size_text = "unlimited"
    else:
        max_file_size = MAX_FILE_SIZE_PRIVATE if has_private_credentials else MAX_FILE_SIZE_REGULAR
        max_file_size_text = "3GB" if has_private_credentials else "300MB"

    if file_size is not None and max_file_size is not None and file_size > max_file_size:
        return None, JSONResponse({
            "error": "errorFileTooLarge",
            "i18n_key": "errorFileTooLarge",
            "i18n_vars": {"size": max_file_size_text}
        }, status_code=400)

    save_audio_bool = str(save_audio).lower() == "true"

    # Block RunPod tokens in local mode
    if in_local_mode and normalized_runpod_token:
        return None, JSONResponse({"error": "errorRunpodNotAvailableInLocalMode", "i18n_key": "errorRunpodNotAvailableInLocalMode"}, status_code=400)

    # Check RunPod endpoint if token provided (only in non-local mode)
    if not in_local_mode and normalized_runpod_token:
        endpoint_result = await check_runpod_endpoint(normalized_runpod_token)
        if not endpoint_result["success"]:
            return None, JSONResponse({
                "error": "errorEndpointCheckFailed",
                "i18n_key": "errorEndpointCheckFailed",
                "i18n_vars": {"details": endpoint_result.get('error', '')}
            }, status_code=400)

        if endpoint_result.get("needs_wait", False):
            action = endpoint_result.get("action", "updated")
            if action == "created":
                i18n_key = "errorEndpointCreated"
            else:
                i18n_key = "errorEndpointUpdated"
            return None, JSONResponse({"error": i18n_key, "i18n_key": i18n_key}, status_code=400)

    metadata = {
        "lang_cfg": lang_cfg,
        "has_private_credentials": has_private_credentials,
        "max_file_size": max_file_size,
        "max_file_size_text": max_file_size_text,
        "save_audio_bool": save_audio_bool,
        "normalized_runpod_token": normalized_runpod_token,
    }

    return metadata, None


@app.post("/upload/precheck", dependencies=[Depends(require_google_login)])
async def precheck_upload(request: Request):
    user_email = get_user_email(request)
    if not user_email:
        return JSONResponse({"error": "User email not found"}, status_code=401)

    session_id = get_session_id(request)
    refresh_token = sessions.get(session_id, {}).get("refresh_token")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "errorInvalidJson", "i18n_key": "errorInvalidJson"}, status_code=400)

    file_size_raw = body.get("file_size")
    if file_size_raw is None:
        return JSONResponse({"error": "errorMissingFileSize", "i18n_key": "errorMissingFileSize"}, status_code=400)

    try:
        file_size = int(file_size_raw)
        if file_size < 0:
            raise ValueError
    except (TypeError, ValueError):
        return JSONResponse({"error": "errorInvalidFileSize", "i18n_key": "errorInvalidFileSize"}, status_code=400)

    metadata, error_response = await validate_upload_request_metadata(
        request,
        user_email=user_email,
        language=body.get("language"),
        runpod_token=body.get("runpod_token"),
        save_audio=body.get("save_audio"),
        file_size=file_size,
        refresh_token=refresh_token,
    )
    if error_response:
        return error_response

    response_payload = {
        "ok": True,
        "max_file_size": metadata["max_file_size"],
        "max_file_size_text": metadata["max_file_size_text"],
        "has_private_credentials": metadata["has_private_credentials"],
    }
    return JSONResponse(response_payload)


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
    global stats_total_jobs_started, stats_total_minutes_processed
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

        # Resolve model by language from config
        lang_cfg = APP_CONFIG["languages"][job_desc.language]
        
        # Select appropriate model based on mode (local vs server)
        if in_local_mode:
            # Use ggml_model for local mode
            selected_model = lang_cfg["ggml_model"]

            # If model is a relative path, resolve it against models_dir
            if args.models_dir and not os.path.isabs(selected_model):
                selected_model = os.path.join(args.models_dir, selected_model)

            log_message(f"{job_desc.user_email}: using local model: {selected_model}")
            m = ivrit.load_model(engine='whisper-cpp', model=selected_model)
            # In local mode, transcribe first without diarization
            segs = m.transcribe_async(path=temp_file_path, diarize=False)

            # Collect all segments for diarization
            all_segments = []
            async for segment in segs:
                all_segments.append(segment)

            # Perform diarization separately after transcription
            from ivrit.diarization import diarize as diarize_func
            import torch
            diarize_device = "cuda" if torch.cuda.is_available() else "cpu"
            diarized_segments = diarize_func(
                audio=temp_file_path,
                transcription_segments=all_segments,
                verbose=True,
                engine="ivrit",
                device=diarize_device
            )

            # Convert to async generator for consistent processing
            async def to_async_generator(segments):
                for segment in segments:
                    yield segment

            segs = to_async_generator(diarized_segments)
        else:
            # Use ct2_model for server/RunPod mode
            selected_model = lang_cfg["ct2_model"]
            
            log_message(f"{job_desc.user_email}: using server model: {selected_model}")
            
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
            
            m = ivrit.load_model(engine='runpod', model=selected_model, api_key=api_key, endpoint_id=endpoint_id, core_engine='stable-whisper')
            
            # Process streaming results
            if in_dev:
                # In dev mode, use the local file
                segs = m.transcribe_async(path=temp_file_path, diarize=True)
            else:
                # In production mode, send file as URL
                base_url = os.environ["BASE_URL"]
                download_url = urljoin(base_url, f"/download/{job_id}")
                segs = m.transcribe_async(url=download_url, diarize=True)
        
        try:
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

        # Update global statistics
        async with stats_lock:
            stats_jobs_transcribed[job_desc.job_type] += 1
            stats_minutes_transcribed[job_desc.job_type] += duration / 60.0  # Convert to minutes
            stats_total_jobs_started += 1
            stats_total_minutes_processed += duration / 60.0

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
            user_identifier = get_user_identifier(refresh_token=job_desc.refresh_token, user_email=job_desc.user_email)
            upload_success = await file_storage_backend.upload_file(
                results_filename,
                file_data,
                mime_type,
                user_identifier,
                job_desc.user_email
            )

            if not upload_success:
                async with stats_lock:
                    stats_gdrive_errors["audio_upload"] += 1
                logger.error(f"Failed to upload results file for {job_id}, skipping TOC update")
                return
            
            # Upload opus file if save_audio is True
            if job_desc.save_audio:
                try:
                    opus_file_path = temp_files.get(job_id)
                    if opus_file_path and os.path.exists(opus_file_path):
                        # Read the opus file
                        with open(opus_file_path, 'rb') as f:
                            opus_data = f.read()
                        
                        # Upload as uuid4.opus
                        opus_filename = f"{results_id}.opus"
                        opus_mime_type = "audio/opus"
                        opus_upload_success = await file_storage_backend.upload_file(
                            opus_filename,
                            opus_data,
                            opus_mime_type,
                            user_identifier,
                            job_desc.user_email
                        )
                        
                        if opus_upload_success:
                            log_message(f"{job_desc.user_email}: Uploaded opus file for {job_id} as {opus_filename}")
                        else:
                            async with stats_lock:
                                stats_gdrive_errors["audio_upload"] += 1
                            logger.warning(f"Failed to upload opus file for {job_id}, but continuing")
                    else:
                        logger.warning(f"Opus file not found for {job_id} at {opus_file_path}, skipping audio upload")
                except Exception as e:
                    logger.error(f"Error uploading opus file for {job_id}: {e}")
                        
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
                session_id_for_toc = None  # We don't have session_id in job_desc, but user_email should be enough for local mode
                toc_data = await download_toc(job_desc.refresh_token, user_email=job_desc.user_email, session_id=session_id_for_toc)

                # Append new entry
                if "entries" not in toc_data:
                    toc_data["entries"] = []
                toc_data["entries"].append(toc_entry)

                # Upload updated TOC atomically
                success = await upload_toc(job_desc.refresh_token, toc_data, user_email=job_desc.user_email, session_id=session_id_for_toc)
                if not success:
                    async with stats_lock:
                        stats_gdrive_errors["toc_upload"] += 1
                    raise Exception("Failed to upload TOC")

            log_message(f"{job_desc.user_email}: Uploaded transcription to TOC (results_id: {results_id})")
        except Exception as e:
            async with stats_lock:
                stats_gdrive_errors["toc_download" if "download" in str(e).lower() else "toc_upload"] += 1
            logger.error(f"Failed to upload transcription for {job_id}: {e}")
    except Exception as e:
        log_message(f"Error in transcription job {job_id}: {str(e)}")
        await emit_upload_error(job_id, "errorInternalServer", details=str(e))
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
        if user_email in user_jobs:
            user_jobs[user_email].discard(job_id)
            if not user_jobs[user_email]:
                del user_jobs[user_email]
        cleanup_temp_file(job_id)
        # The job thread will terminate itself in the next iteration of the transcribe_job function


async def submit_next_task(job_queue, running_jobs, max_parallel_jobs, queue_type):
    async with queue_locks[queue_type]:
        # Build per-user running count from current running jobs
        user_running_count = {}
        for j in running_jobs.values():
            user_running_count[j.user_email] = user_running_count.get(j.user_email, 0) + 1

        deferred = []
        while len(running_jobs) < max_parallel_jobs and not job_queue.empty():
            job_desc = job_queue.get()
            if user_running_count.get(job_desc.user_email, 0) < MAX_PARALLEL_JOBS_PER_USER:
                running_jobs[job_desc.id] = job_desc
                user_running_count[job_desc.user_email] = user_running_count.get(job_desc.user_email, 0) + 1
                asyncio.create_task(transcribe_job(job_desc))
            else:
                deferred.append(job_desc)
        for job_desc in deferred:
            job_queue.put(job_desc)


async def submit_next_transcoding_task():
    """Submit next transcoding task from queue if capacity available"""
    async with transcoding_lock:
        # Submit all possible tasks until max_parallel_jobs are reached or queue is empty
        while len(transcoding_running_jobs) < MAX_PARALLEL_TRANSCODES and not transcoding_queue.empty():
            transcoding_job = transcoding_queue.get()
            # Track start time for ETA calculation
            transcoding_job.transcode_start_time = time.time()
            transcoding_running_jobs[transcoding_job.id] = transcoding_job
            # Create async task for transcoding
            asyncio.create_task(handle_transcoding(transcoding_job.id))


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




async def check_heartbeat_timeout():
    """Check if heartbeat has timed out in local mode and shutdown if needed."""
    global missed_heartbeat_count, heartbeat_received_this_period, last_heartbeat_check_time
    
    if not in_local_mode:
        return
    
    current_time = time.time()
    
    # Initialize last check time on first call
    if last_heartbeat_check_time is None:
        last_heartbeat_check_time = current_time
        return
    
    # Only check once per interval
    if current_time - last_heartbeat_check_time < HEARTBEAT_INTERVAL_SECONDS:
        return
    
    last_heartbeat_check_time = current_time
    
    # Check if heartbeat was received during this period
    if heartbeat_received_this_period:
        # Heartbeat received, reset counter
        missed_heartbeat_count = 0
        heartbeat_received_this_period = False
    else:
        # No heartbeat received, increment missed count
        missed_heartbeat_count += 1
        log_message(f"Heartbeat missed ({missed_heartbeat_count}/{HEARTBEAT_MISSED_THRESHOLD})")
    
    if missed_heartbeat_count >= HEARTBEAT_MISSED_THRESHOLD:
        log_message(f"Auto-shutdown: {missed_heartbeat_count} consecutive heartbeats missed (threshold: {HEARTBEAT_MISSED_THRESHOLD})")
        log_message("Shutting down server due to client inactivity...")
        # Give a moment for the log to be written
        await asyncio.sleep(0.5)
        os._exit(0)


async def event_loop():
    while True:
        await submit_next_transcoding_task()
        await submit_next_task(queues[SHORT], running_jobs[SHORT], max_parallel_jobs[SHORT], SHORT)
        await submit_next_task(queues[LONG], running_jobs[LONG], max_parallel_jobs[LONG], LONG)
        await submit_next_task(queues[PRIVATE], running_jobs[PRIVATE], max_parallel_jobs[PRIVATE], PRIVATE)
        await cleanup_old_results()
        await check_heartbeat_timeout()
        await asyncio.sleep(0.1)


# Global async resources
queue_locks = {
    SHORT: None,
    LONG: None,
    PRIVATE: None
}

def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    port = 4600 if in_dev else 4500

    # Check if port is available before starting
    if not check_port_available(port):
        logger.error(f"Port {port} is already in use. Another instance of the server may be running.")
        logger.error("Please close the other instance or use a different port.")
        print(f"\n*** ERROR: Port {port} is already in use. ***")
        print("Another instance of the transcription server may be running.")
        print("Please close it before starting a new one.\n")
        sys.exit(1)

    # Configure SSL if dev-https is enabled
    ssl_kwargs = {}
    if args.dev_https:
        if not args.dev_cert_folder:
            raise RuntimeError("--dev-cert-folder is required when --dev-https is enabled")

        cert_file = os.path.join(args.dev_cert_folder, "cert.pem")
        key_file = os.path.join(args.dev_cert_folder, "key.pem")

        if not os.path.exists(cert_file):
            raise RuntimeError(f"Certificate file not found: {cert_file}")
        if not os.path.exists(key_file):
            raise RuntimeError(f"Key file not found: {key_file}")

        ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}

    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None, **ssl_kwargs)
