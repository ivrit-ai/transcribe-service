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
import httpx
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

import traceback

import posthog
import ffmpeg
import base64
import argparse
import re
import ivrit

dotenv.load_dotenv()

# Parse CLI arguments for configuration
parser = argparse.ArgumentParser(description='Transcription service with rate limiting')
parser.add_argument('--max-minutes-per-week', type=int, default=180, help='Maximum credit grant in minutes per week (default: 420)')
parser.add_argument('--staging', action='store_true', help='Enable staging mode')
parser.add_argument('--hiatus', action='store_true', help='Enable hiatus mode')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('--dev', action='store_true', help='Enable development mode')
parser.add_argument('--dev-user-email', help='User email for development mode')
args, unknown = parser.parse_known_args()

# Rate limiting configuration from CLI arguments
MAX_MINUTES_PER_WEEK = args.max_minutes_per_week  # Maximum credit grant per week
REPLENISH_RATE_MINUTES_PER_DAY = MAX_MINUTES_PER_WEEK / 7  # Automatically derive daily replenish rate



in_dev = args.staging or args.dev
in_hiatus_mode = args.hiatus
verbose = args.verbose

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

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "https://transcribe.ivrit.ai/login/authorized")

def log_message_in_session(message, user_email=None):
    if user_email:
        logger.info(f"{user_email}: {message}")
    else:
        logger.info(f"{message}")

def log_message(message):
    logger.info(f"{message}")

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
JOB_TIMEOUT = 1 * 60

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

# Dictionary to keep track of last access time for each job
job_last_accessed = {}

# Dictionary to store user rate limiting buckets
user_buckets = {}





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


async def calculate_queue_time(queue_to_use, running_jobs, exclude_last=False):
    """
    Calculate the estimated time remaining for jobs in queue and running jobs

    Args:
        queue_to_use: Queue to check
        running_jobs: Dictionary of currently running jobs
        exclude_last: Whether to exclude the last job in queue from calculation

    Returns:
        time_ahead_str: Formatted string of estimated time (HH:MM:SS)
    """
    # Skip queue time calculations for private queue
    if queue_to_use == queues[PRIVATE]:
        return "00:00:00"
    
    # Determine which lock to use based on the queue
    queue_type = None
    for qt, q in queues.items():
        if q == queue_to_use:
            queue_type = qt
            break
    
    if queue_type is None:
        return "00:00:00"
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


async def queue_job(job_id, user_email, filename, duration, runpod_token=""):
    # Try to add the job to the queue
    log_message_in_session(f"Queuing job {job_id}...")

    # Check if user already has a job queued or running
    if user_email in user_jobs:
        return False, JSONResponse({"error": "יש לך כבר עבודה בתור או בביצוע. אנא המתן לסיומה לפני העלאת קובץ חדש."}, status_code=400)

    # Check rate limits only if not using custom RunPod credentials
    custom_runpod_credentials = bool(runpod_token)
    if not custom_runpod_credentials:
        user_bucket = get_user_quota(user_email)
        if not user_bucket.can_transcribe(duration):
            remaining_minutes = user_bucket.get_remaining_minutes()
            log_message_in_session(f"Job queuing rate limited for user {user_email}. Requested: {duration/60:.1f}min, Remaining: {remaining_minutes:.1f}min")
            
            # Calculate how many minutes they need to wait, accounting for replenish rate
            needed_minutes = (duration / 60) - remaining_minutes
            replenish_rate_per_minute = user_bucket.time_fill_rate * 60
            wait_minutes = math.ceil(needed_minutes / (1 + replenish_rate_per_minute))

            # Convert time_fill_rate (per second) to minutes per minute
            error_msg = f"עברת את מגבלת השימוש החופשי. אנא המתן {wait_minutes} דקות לפני העלאת קובץ חדש, או השתמש במפתח פרטי בעזרת ההוראות בסרטון הבא: https://www.youtube.com/watch?v=mEgfmO7zzdw"
            
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
            time_ahead_str = await calculate_queue_time(queue_to_use, running_jobs_to_update, exclude_last=True)

            # Add job to user's active jobs
            user_jobs[user_email] = job_id

            # Set initial access time
            job_last_accessed[job_id] = time.time()

            capture_event(job_id, "job-queued", {"user": user_email, "queue-depth": queue_depth, "job-type": job_type, "custom-runpod": job_desc.uses_custom_runpod})

            log_message_in_session(
                f"Job queued successfully: {job_id}, queue depth: {queue_depth}, job type: {job_type}, job desc: {job_desc}"
            )

            return True, (
                JSONResponse({"job_id": job_id, "queued": queue_depth, "job_type": job_type, "time_ahead": time_ahead_str})
            )
    except queue.Full:
        capture_event(job_id, "job-queue-failed", {"queue-depth": queue_depth, "job-type": job_type})

        log_message_in_session(f"Job queuing failed: {job_id}")

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
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": state
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    
    response = RedirectResponse(url=auth_url)
    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=not in_dev)
    return response

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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GRAPHQL_URL, 
                headers=GRAPHQL_HEADERS, 
                json={"query": query},
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL Error: {data['errors']}")
                return None
                
            balance_info = data["data"]["myself"]
            return {
                "clientBalance": balance_info.get('clientBalance', 'N/A'),
                "currentSpendPerHr": balance_info.get('currentSpendPerHr', 'N/A'),
                "spendLimit": balance_info.get('spendLimit', 'N/A')
            }
            
    except httpx.RequestError as e:
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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GRAPHQL_URL, 
                headers=GRAPHQL_HEADERS, 
                json={"query": query},
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL Error: {data['errors']}")
                return None
                
            endpoints = data["data"]["myself"]["endpoints"]
            
            # Look for endpoints matching the pattern autogenerated-endpoint-transcribe-ivrit-ai-*
            pattern_regex = r"autogenerated-endpoint-transcribe-ivrit-ai-\d{8}"
            for endpoint in endpoints:
                if re.match(pattern_regex, endpoint["name"]):
                    logger.info(f"Found endpoint: {endpoint['id']} ({endpoint['name']}) with template {endpoint['templateId']}")
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
            
    except httpx.RequestError as e:
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
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{REST_URL}/endpoints/{endpoint_id}",
                headers=REST_HEADERS,
                timeout=30.0
            )
            response.raise_for_status()
            
            logger.info(f"Successfully deleted endpoint: {endpoint_id}")
            return True
            
    except httpx.RequestError as e:
        logger.error(f"Error deleting endpoint {endpoint_id}: {e}")
        return False

async def create_runpod_endpoint(api_key: str, template_id: str) -> Optional[dict]:
    """Create a new autogenerated endpoint using REST API"""
    REST_URL = "https://rest.runpod.io/v1"
    REST_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Generate date-based name
    current_date = datetime.now().strftime("%Y%m%d")
    endpoint_name = f"autogenerated-endpoint-transcribe-ivrit-ai-{current_date}"
    
    endpoint_data = {
        "name": endpoint_name,
        "templateId": template_id,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": 3,
        "idleTimeout": 5,
        "executionTimeoutMs": 600000
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{REST_URL}/endpoints",
                headers=REST_HEADERS,
                json=endpoint_data,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            endpoint_id = result.get('id')
            
            logger.info(f"Created autogenerated endpoint: {endpoint_id} ({endpoint_name})")
            return {
                "id": endpoint_id,
                "name": endpoint_name,
                "status": result.get('status', 'N/A')
            }
            
    except httpx.RequestError as e:
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

@app.post("/check_endpoint")
async def check_endpoint(request: Request):
    """Check for autogenerated endpoint, validate template ID, and recreate if needed"""
    try:
        body = await request.json()
        runpod_token = body.get("runpod_token")
        
        if not runpod_token:
            return JSONResponse({"error": "Missing RunPod token"}, status_code=400)
        
        # Get the required template ID from environment
        required_template_id = os.environ.get("RUNPOD_TEMPLATE_ID")
        if not required_template_id:
            return JSONResponse({"error": "RUNPOD_TEMPLATE_ID environment variable not set"}, status_code=500)
        
        # Check if endpoint exists with full details
        endpoint_info = await find_runpod_endpoint(runpod_token)
        
        if endpoint_info:
            # Endpoint found, check if template ID matches
            current_template_id = endpoint_info.get("templateId")
            
            if current_template_id == required_template_id:
                # Template ID matches - endpoint is up to date
                return JSONResponse({
                    "action": "up_to_date",
                    "endpoint_id": endpoint_info["id"],
                    "endpoint_name": endpoint_info["name"],
                    "template_id": current_template_id
                })
            else:
                # Template ID doesn't match - delete and recreate
                logger.info(f"Template ID mismatch: current={current_template_id}, required={required_template_id}")
                
                # Delete the existing endpoint
                delete_success = await delete_runpod_endpoint(runpod_token, endpoint_info["id"])
                if not delete_success:
                    logger.warning(f"Failed to delete endpoint {endpoint_info['id']}, but proceeding with creation")
                
                # Create new endpoint with correct template
                new_endpoint = await create_runpod_endpoint(runpod_token, required_template_id)
                
                if new_endpoint:
                    return JSONResponse({
                        "action": "updated",
                        "endpoint_id": new_endpoint["id"],
                        "endpoint_name": new_endpoint["name"],
                        "old_template_id": current_template_id,
                        "new_template_id": required_template_id,
                        "status": new_endpoint["status"]
                    })
                else:
                    return JSONResponse({"error": "Failed to create updated endpoint"}, status_code=500)
        else:
            # No endpoint found - create a new one
            new_endpoint = await create_runpod_endpoint(runpod_token, required_template_id)
            
            if new_endpoint:
                return JSONResponse({
                    "action": "created",
                    "endpoint_id": new_endpoint["id"],
                    "endpoint_name": new_endpoint["name"],
                    "template_id": required_template_id,
                    "status": new_endpoint["status"]
                })
            else:
                return JSONResponse({"error": "Failed to create endpoint"}, status_code=500)
            
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
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                }
            )
            token_data = token_response.json()
            
            if "error" in token_data:
                error_message = f"Token exchange failed: {token_data.get('error_description', 'Unknown error')}"
                return templates.TemplateResponse("close_window.html", {"request": request, "success": False, "message": error_message})
            
            access_token = token_data["access_token"]
            
            # Get user info using v2 endpoint
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_response.json()
            
            # Store user email in session
            set_user_email(request, user_data["email"])
            
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
    runpod_token: Optional[str] = Form(None)
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

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    try:
        # Read file content and save to temp file
        content = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        return JSONResponse({"error": f"העלאת הקובץ נכשלה: {str(e)}"}, status_code=200)

    # Get the MIME type of the file
    filetype = magic.Magic(mime=True).from_file(temp_file_path)

    if not is_ffmpeg_supported_mimetype(filetype):
        return JSONResponse({"error": f"סוג הקובץ {filetype} אינו נתמך. אנא העלה קובץ אודיו או וידאו תקין."}, status_code=200)

    # Get the duration of the audio file
    duration = get_media_duration(temp_file_path)

    if duration is None:
        return JSONResponse({"error": "לא ניתן לקרוא את משך הקובץ. אנא ודא שהקובץ תקין ונסה שוב."}, status_code=200)

    # Get RunPod token if provided
    runpod_token = runpod_token.strip() if runpod_token else ""
    
    # Store the temporary file path
    temp_files[job_id] = temp_file_path

    # Use the background event loop to call the async function
    queued, res = await queue_job(job_id, user_email, filename, duration, runpod_token)
    if queued:
        job_results[job_id] = {"results": [], "completion_time": None, "progress": 0}
    else:
        cleanup_temp_file(job_id)

    return res


@app.get("/job_status/{job_id}")
async def job_status(job_id: str, request: Request):
    # Make sure the job has been queued
    if job_id not in job_results:
        return JSONResponse({"error": "מזהה העבודה אינו תקין. אנא נסה להעלות את הקובץ מחדש."}, status_code=400)

    # Update last access time for the job
    job_last_accessed[job_id] = time.time()

    # Check if the job is in the queue
    queue_position = None
    job_type = None
    queue_to_use = None

    # Check all queues for the job
    for queue_type in [SHORT, LONG, PRIVATE]:
        for idx, job_desc in enumerate(queues[queue_type].queue):
            if job_desc.id == job_id:
                queue_position = idx + 1
                job_type = queue_type
                queue_to_use = queues[queue_type]
                break
        if queue_position:
            break

    if queue_position:
        running_jobs_to_check = running_jobs[job_type]
        # Use the background event loop to call the async function
        time_ahead_str = await calculate_queue_time(queue_to_use, running_jobs_to_check)
        return JSONResponse({"queue_position": queue_position, "time_ahead": time_ahead_str, "job_type": job_type})

    # If job is in progress, return only the progress
    if job_results[job_id]["progress"] < 1.0:
        return JSONResponse({"progress": job_results[job_id]["progress"]})

    # If job is complete, return the full job_result data
    result = job_results[job_id].copy()
    # Convert datetime to ISO string for JSON serialization
    if result["completion_time"]:
        result["completion_time"] = result["completion_time"].isoformat()
    return JSONResponse(result)


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
    # Check if the job should be terminated
    if time.time() - job_last_accessed.get(job_id, 0) > JOB_TIMEOUT:
        log_message(f"Terminating inactive job: {job_id}")
        return False

    segment_data = {
        "id": segment.extra_data.get("id"),
        "start": segment.start,
        "end": segment.end,
        "text": clean_some_unicode_from_text(segment.text),
        "avg_logprob": segment.extra_data.get("avg_logprob"),
        "compression_ratio": segment.extra_data.get("compression_ratio"),
        "no_speech_prob": segment.extra_data.get("no_speech_prob"),
    }

    progress = segment.end / duration
    job_results[job_id]["progress"] = progress
    job_results[job_id]["results"].append(segment_data)

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
        
        m = ivrit.load_model(engine='runpod', model='ivrit-ai/whisper-large-v3-turbo-ct2', api_key=api_key, endpoint_id=endpoint_id)
        
        # Process streaming results
        try:
            if in_dev:
                # In dev mode, use the local file
                segs = m.transcribe_async(path=temp_file_path, stream=True)
            else:
                # In production mode, send file as URL
                base_url = os.environ["BASE_URL"]
                download_url = urljoin(base_url, f"/download/{job_id}")
                segs = m.transcribe_async(url=download_url, stream=True)
            
            async for segment in segs:
                if not await process_segment(job_id, segment, duration):
                    # Job was terminated
                    log_message(f"Terminating runpod job due to process_segment error.")
                    return

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
                "transcription_seconds": transcribe_done_time - transcribe_start_time,
                "audio_duration_seconds": duration,
                "job-type": job_desc.job_type,
                "custom-runpod": job_desc.uses_custom_runpod,
            },
        )

        job_results[job_id]["progress"] = 1.0
        job_results[job_id]["completion_time"] = datetime.now()
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


async def terminate_inactive_jobs():
    current_time = time.time()
    jobs_to_terminate = []

    # Check queued jobs in all queues.
    # Running jobs are handled in transcribe_job.
    for queue_type in [SHORT, LONG, PRIVATE]:
        async with queue_locks[queue_type]:
            queue_to_check = queues[queue_type]
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


async def event_loop():
    while True:
        await submit_next_task(queues[SHORT], running_jobs[SHORT], max_parallel_jobs[SHORT], SHORT)
        await submit_next_task(queues[LONG], running_jobs[LONG], max_parallel_jobs[LONG], LONG)
        await submit_next_task(queues[PRIVATE], running_jobs[PRIVATE], max_parallel_jobs[PRIVATE], PRIVATE)
        await cleanup_old_results()
        await terminate_inactive_jobs()
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
