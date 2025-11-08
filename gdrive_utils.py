"""Google Drive utilities for appDataFolder operations."""
import os
import json
import gzip
import logging
import aiohttp
import dotenv
from typing import Optional

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "https://transcribe.ivrit.ai/login/authorized")


async def refresh_google_access_token(refresh_token: str) -> Optional[dict]:
    """Use a refresh token to obtain a new access token."""
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
                data = await resp.json()
                if "error" in data:
                    logger.error(f"Google token refresh failed: {data}")
                    return None
                return data
    except Exception as e:
        logger.error(f"Exception during token refresh: {e}")
        return None


async def get_access_token_from_refresh(refresh_token: Optional[str]) -> Optional[str]:
    """Refresh and return a new Google access token using the provided refresh token."""
    if not refresh_token:
        return None
    refreshed = await refresh_google_access_token(refresh_token)
    if not refreshed:
        return None
    return refreshed.get("access_token")


async def upload_to_google_appdata(refresh_token: Optional[str], filename: str, file_data: bytes, mime_type: str, user_email: Optional[str] = None) -> bool:
    """Upload a file into the user's Drive appDataFolder using a refresh token.
    
    Args:
        refresh_token: Google refresh token
        filename: Name of the file to upload
        file_data: Raw bytes to upload (caller should compress if needed)
        mime_type: MIME type of the file (e.g., "application/json" or "application/gzip")
        user_email: Optional user email for logging
    
    Returns:
        True if successful, False otherwise
    """
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning(f"No Google access token available; skipping appData upload for {user_email or 'unknown user'}")
        return False
    try:
        form = aiohttp.FormData()
        metadata = {
            "name": filename,
            "mimeType": mime_type,
            "parents": ["appDataFolder"],
        }
        form.add_field("metadata", json.dumps(metadata), content_type="application/json; charset=UTF-8")
        form.add_field("file", file_data, content_type=mime_type)

        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                data=form,
                headers=headers,
            ) as resp:
                if resp.status >= 200 and resp.status < 300:
                    logger.info(f"Uploaded file to appData for {user_email} as {filename}")
                    return True
                err = await resp.text()
                logger.error(f"Failed to upload to appData for {user_email}: {resp.status} {err}")
                return False
    except Exception as e:
        logger.error(f"Exception uploading to appData for {user_email}: {e}")
        return False


async def update_google_appdata_file(refresh_token: Optional[str], file_id: str, file_data: bytes, mime_type: str, user_email: Optional[str] = None) -> bool:
    """Update an existing file in appDataFolder with new content.
    
    Args:
        refresh_token: Google refresh token
        file_id: Google Drive file ID
        file_data: Raw bytes to upload (caller should compress if needed)
        mime_type: MIME type of the file (e.g., "application/json" or "application/gzip")
        user_email: Optional user email for logging
    
    Returns:
        True if successful, False otherwise
    """
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning(f"No Google access token available; skipping appData update for {user_email or 'unknown user'}")
        return False
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": mime_type,
        }
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                f"https://www.googleapis.com/upload/drive/v3/files/{file_id}?uploadType=media",
                headers=headers,
                data=file_data,
            ) as resp:
                if resp.status >= 200 and resp.status < 300:
                    logger.info(f"Updated file {file_id} in appData for {user_email}")
                    return True
                err = await resp.text()
                logger.error(f"Failed to update file {file_id} for {user_email}: {resp.status} {err}")
                return False
    except Exception as e:
        logger.error(f"Exception updating file {file_id} for {user_email}: {e}")
        return False


async def download_google_appdata_file_bytes(refresh_token: Optional[str], file_id: str) -> Optional[bytes]:
    """Download a specific file from the user's Drive appDataFolder as raw bytes.
    
    Args:
        refresh_token: Google refresh token
        file_id: Google Drive file ID
    
    Returns:
        Raw bytes or None on error
    """
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning("No Google access token available for downloading appData file")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                headers=headers,
            ) as resp:
                if resp.status >= 200 and resp.status < 300:
                    return await resp.read()
                err = await resp.text()
                logger.error(f"Failed to download appData file {file_id}: {resp.status} {err}")
                return None
    except Exception as e:
        logger.error(f"Exception downloading appData file {file_id}: {e}")
        return None


async def list_google_appdata_files(refresh_token: Optional[str]) -> Optional[list]:
    """List files in the user's Drive appDataFolder."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning("No Google access token available for listing appData files")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/drive/v3/files?spaces=appDataFolder&fields=files(id,name,createdTime,modifiedTime,size)",
                headers=headers,
            ) as resp:
                if resp.status >= 200 and resp.status < 300:
                    data = await resp.json()
                    return data.get("files", [])
                err = await resp.text()
                logger.error(f"Failed to list appData files: {resp.status} {err}")
                return None
    except Exception as e:
        logger.error(f"Exception listing appData files: {e}")
        return None


async def find_google_appdata_file_by_name(refresh_token: Optional[str], filename: str) -> Optional[str]:
    """Find a file in appDataFolder by name and return its ID."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://www.googleapis.com/drive/v3/files?spaces=appDataFolder&q=name='{filename}'&fields=files(id)",
                headers=headers,
            ) as resp:
                if resp.status >= 200 and resp.status < 300:
                    data = await resp.json()
                    files = data.get("files", [])
                    if files:
                        return files[0]["id"]
                    return None
                return None
    except Exception as e:
        logger.error(f"Exception finding file by name {filename}: {e}")
        return None

