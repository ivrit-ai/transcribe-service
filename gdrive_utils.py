"""Google Drive utilities for managing Drive folder storage."""

import os
import json
import hashlib
import logging
import aiohttp
import dotenv
from typing import Optional, Dict

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger("transcribe_service.gdrive_utils")

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "https://transcribe.ivrit.ai/login/authorized")

DRIVE_FOLDER_NAME = os.environ.get("GOOGLE_DRIVE_FOLDER_NAME", "transcribe.ivrit.ai")
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

# Cache folder IDs per refresh token to avoid repeated Drive lookups within a process lifetime
folder_id_cache: Dict[str, str] = {}


def _get_folder_cache_key(refresh_token: str) -> str:
    return hashlib.sha256(refresh_token.encode()).hexdigest()


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


async def ensure_drive_folder(refresh_token: Optional[str], access_token: Optional[str] = None) -> Optional[str]:
    """Ensure the application Drive folder exists and return its ID."""
    if not refresh_token:
        return None

    cache_key = _get_folder_cache_key(refresh_token)
    cached = folder_id_cache.get(cache_key)
    if cached:
        return cached

    token = access_token or await get_access_token_from_refresh(refresh_token)
    if not token:
        logger.warning("No Google access token available; cannot ensure Drive folder")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
    }
    params = {
        "q": f"name = '{DRIVE_FOLDER_NAME}' and mimeType = '{FOLDER_MIME_TYPE}' and trashed = false",
        "spaces": "drive",
        "fields": "files(id,name)",
        "pageSize": 1,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/drive/v3/files",
                headers=headers,
                params=params,
            ) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.json()
                    files = data.get("files", [])
                    if files:
                        folder_id = files[0]["id"]
                        folder_id_cache[cache_key] = folder_id
                        return folder_id
                else:
                    err = await resp.text()
                    logger.error(f"Failed to query Drive folder: {resp.status} {err}")
                    return None

            create_headers = {
                "Authorization": headers["Authorization"],
                "Content-Type": "application/json",
            }
            metadata = {
                "name": DRIVE_FOLDER_NAME,
                "mimeType": FOLDER_MIME_TYPE,
            }
            async with session.post(
                "https://www.googleapis.com/drive/v3/files",
                headers=create_headers,
                params={"fields": "id"},
                json=metadata,
            ) as create_resp:
                if 200 <= create_resp.status < 300:
                    data = await create_resp.json()
                    folder_id = data.get("id")
                    if folder_id:
                        folder_id_cache[cache_key] = folder_id
                        logger.info(f"Created Drive folder '{DRIVE_FOLDER_NAME}' with id {folder_id}")
                        return folder_id
                    logger.error("Drive API response missing folder id after creation")
                    return None
                err = await create_resp.text()
                logger.error(f"Failed to create Drive folder: {create_resp.status} {err}")
                return None
    except Exception as e:
        logger.error(f"Exception ensuring Drive folder: {e}")
        return None


async def upload_to_drive_folder(refresh_token: Optional[str], filename: str, file_data: bytes, mime_type: str, user_email: Optional[str] = None) -> bool:
    """Upload a file into the application's Drive folder using the drive.file scope."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning(f"No Google access token available; skipping Drive upload for {user_email or 'unknown user'}")
        return False

    folder_id = await ensure_drive_folder(refresh_token, access_token)
    if not folder_id:
        logger.error("Unable to determine Drive folder id for upload")
        return False

    try:
        form = aiohttp.FormData()
        metadata = {
            "name": filename,
            "mimeType": mime_type,
            "parents": [folder_id],
        }
        form.add_field("metadata", json.dumps(metadata), content_type="application/json; charset=UTF-8")
        form.add_field("file", file_data, content_type=mime_type)

        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://www.googleapis.com/upload/drive/v3/files",
                params={"uploadType": "multipart", "fields": "id"},
                data=form,
                headers=headers,
            ) as resp:
                if 200 <= resp.status < 300:
                    logger.info(f"Uploaded file to Drive for {user_email} as {filename}")
                    return True
                err = await resp.text()
                logger.error(f"Failed to upload to Drive for {user_email}: {resp.status} {err}")
                return False
    except Exception as e:
        logger.error(f"Exception uploading to Drive for {user_email}: {e}")
        return False


async def update_drive_file(refresh_token: Optional[str], file_id: str, file_data: bytes, mime_type: str, user_email: Optional[str] = None) -> bool:
    """Update an existing file in the application's Drive folder."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning(f"No Google access token available; skipping Drive update for {user_email or 'unknown user'}")
        return False
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": mime_type,
        }
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                f"https://www.googleapis.com/upload/drive/v3/files/{file_id}",
                params={"uploadType": "media", "fields": "id"},
                headers=headers,
                data=file_data,
            ) as resp:
                if 200 <= resp.status < 300:
                    logger.info(f"Updated Drive file {file_id} for {user_email}")
                    return True
                err = await resp.text()
                logger.error(f"Failed to update Drive file {file_id} for {user_email}: {resp.status} {err}")
                return False
    except Exception as e:
        logger.error(f"Exception updating Drive file {file_id} for {user_email}: {e}")
        return False


async def download_drive_file_bytes(refresh_token: Optional[str], file_id: str) -> Optional[bytes]:
    """Download a specific file from the application's Drive folder as raw bytes."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning("No Google access token available for downloading Drive file")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                params={"alt": "media"},
                headers=headers,
            ) as resp:
                if 200 <= resp.status < 300:
                    return await resp.read()
                err = await resp.text()
                logger.error(f"Failed to download Drive file {file_id}: {resp.status} {err}")
                return None
    except Exception as e:
        logger.error(f"Exception downloading Drive file {file_id}: {e}")
        return None


async def list_drive_folder_files(refresh_token: Optional[str]) -> Optional[list]:
    """List files stored in the application's Drive folder."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning("No Google access token available for listing Drive files")
        return None

    folder_id = await ensure_drive_folder(refresh_token, access_token)
    if not folder_id:
        return None

    params = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "spaces": "drive",
        "fields": "files(id,name,createdTime,modifiedTime,size)",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/drive/v3/files",
                headers=headers,
                params=params,
            ) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.json()
                    return data.get("files", [])
                err = await resp.text()
                logger.error(f"Failed to list Drive files: {resp.status} {err}")
                return None
    except Exception as e:
        logger.error(f"Exception listing Drive files: {e}")
        return None


async def find_drive_file_by_name(refresh_token: Optional[str], filename: str) -> Optional[str]:
    """Find a file by name within the application's Drive folder and return its ID."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        return None

    folder_id = await ensure_drive_folder(refresh_token, access_token)
    if not folder_id:
        return None

    quoted_filename = filename.replace("'", "\\'")
    params = {
        "q": f"name = '{quoted_filename}' and '{folder_id}' in parents and trashed = false",
        "spaces": "drive",
        "fields": "files(id)",
        "pageSize": 1,
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/drive/v3/files",
                headers=headers,
                params=params,
            ) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.json()
                    files = data.get("files", [])
                    if files:
                        return files[0]["id"]
                    return None
                err = await resp.text()
                logger.error(f"Failed to search Drive file {filename}: {resp.status} {err}")
                return None
    except Exception as e:
        logger.error(f"Exception finding Drive file by name {filename}: {e}")
        return None


async def delete_drive_file(refresh_token: Optional[str], file_id: str, user_email: Optional[str] = None) -> bool:
    """Delete a file from Google Drive by its file ID."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning(f"No Google access token available for deleting Drive file for {user_email or 'unknown user'}")
        return False
    
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                headers=headers,
            ) as resp:
                if resp.status == 204:
                    logger.info(f"Deleted Drive file {file_id} for {user_email or 'unknown user'}")
                    return True
                err = await resp.text()
                logger.warning(f"Failed to delete Drive file {file_id} for {user_email}: {resp.status} {err}")
                return False
    except Exception as e:
        logger.error(f"Exception deleting Drive file {file_id} for {user_email}: {e}")
        return False

