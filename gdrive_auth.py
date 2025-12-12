"""Google Drive authentication and token management utilities."""

import os
import time
import hashlib
import logging
import aiohttp
import dotenv
from typing import Optional, Dict, Any

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger("transcribe_service.gdrive_auth")

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "https://transcribe.ivrit.ai/login/authorized")

# Access token caching
ACCESS_TOKEN_EXPIRY_TIME = int(os.environ.get("GOOGLE_ACCESS_TOKEN_EXPIRY_SECONDS", "3600"))
ACCESS_TOKEN_REFRESH_THRESHOLD = ACCESS_TOKEN_EXPIRY_TIME * 0.9
access_token_cache: Dict[str, Dict[str, Any]] = {}


class GoogleAPIError(Exception):
    """Base exception for Google API interactions."""


class GoogleAuthError(GoogleAPIError):
    """Raised when Google OAuth operations fail."""


class GoogleDriveError(GoogleAPIError):
    """Raised when Google Drive operations fail."""


def _get_token_cache_key(refresh_token: str) -> str:
    """Generate a cache key from a refresh token."""
    return hashlib.sha256(refresh_token.encode()).hexdigest()


async def refresh_google_access_token(refresh_token: str) -> Optional[dict]:
    """Use a refresh token to obtain a new access token, with time-based caching."""
    cache_key = _get_token_cache_key(refresh_token)
    cached_entry = access_token_cache.get(cache_key)

    if cached_entry:
        age = time.time() - cached_entry["timestamp"]
        if age < ACCESS_TOKEN_REFRESH_THRESHOLD:
            return cached_entry["data"]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            ) as resp:
                if not 200 <= resp.status < 300:
                    err = await resp.text()
                    logger.error(
                        "Failed to refresh Google access token: %s %s",
                        resp.status,
                        err,
                    )
                    raise GoogleAuthError(
                        f"Failed to refresh Google access token: {resp.status} {err}"
                    )
                data = await resp.json()
                if "error" in data:
                    logger.error("Google token refresh failed: %s", data)
                    raise GoogleAuthError(f"Google token refresh failed: {data}")

                access_token_cache[cache_key] = {
                    "data": data,
                    "timestamp": time.time(),
                }
                return data
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception during token refresh")
        raise GoogleAuthError("Exception during token refresh") from exc


async def get_access_token_from_refresh(refresh_token: Optional[str]) -> Optional[str]:
    """Refresh and return a new Google access token using the provided refresh token."""
    if not refresh_token:
        return None
    refreshed = await refresh_google_access_token(refresh_token)
    if not refreshed:
        return None
    return refreshed.get("access_token")

