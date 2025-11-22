"""Google Drive utilities for managing Drive folder storage."""

import os
import json
import time
import hashlib
import logging
import aiohttp
import dotenv
from typing import Optional, Dict, Any
# Access token caching
ACCESS_TOKEN_EXPIRY_TIME = int(os.environ.get("GOOGLE_ACCESS_TOKEN_EXPIRY_SECONDS", "3600"))
ACCESS_TOKEN_REFRESH_THRESHOLD = ACCESS_TOKEN_EXPIRY_TIME * 0.9
access_token_cache: Dict[str, Dict[str, Any]] = {}

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger("transcribe_service.gdrive_utils")
logger.info("gdrive_utils module loaded - this message should appear in both console and app.log")

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "https://transcribe.ivrit.ai/login/authorized")

DRIVE_FOLDER_NAME = os.environ.get("GOOGLE_DRIVE_FOLDER_NAME", "transcribe.ivrit.ai")
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

# Cache folder IDs per refresh token to avoid repeated Drive lookups within a process lifetime
folder_id_cache: Dict[str, str] = {}


class GoogleAPIError(Exception):
    """Base exception for Google API interactions."""


class GoogleAuthError(GoogleAPIError):
    """Raised when Google OAuth operations fail."""


class GoogleDriveError(GoogleAPIError):
    """Raised when Google Drive operations fail."""


def _get_folder_cache_key(refresh_token: str) -> str:
    return hashlib.sha256(refresh_token.encode()).hexdigest()


async def refresh_google_access_token(refresh_token: str) -> Optional[dict]:
    """Use a refresh token to obtain a new access token, with time-based caching."""
    cache_key = _get_folder_cache_key(refresh_token)
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
                    logger.error(
                        "Failed to query Drive folder: %s %s",
                        resp.status,
                        err,
                    )
                    raise GoogleDriveError(
                        f"Failed to query Drive folder: {resp.status} {err}"
                    )

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
                        logger.info(
                            "Created Drive folder '%s' with id %s",
                            DRIVE_FOLDER_NAME,
                            folder_id,
                        )
                        return folder_id
                    logger.error("Drive API response missing folder id after creation")
                    raise GoogleDriveError(
                        "Drive API response missing folder id after creation"
                    )
                err = await create_resp.text()
                logger.error(
                    "Failed to create Drive folder: %s %s",
                    create_resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to create Drive folder: {create_resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception ensuring Drive folder")
        raise GoogleDriveError("Exception ensuring Drive folder") from exc


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
                    logger.info(
                        "Uploaded file to Drive for %s as %s",
                        user_email,
                        filename,
                    )
                    return True
                err = await resp.text()
                logger.error(
                    "Failed to upload to Drive for %s: %s %s",
                    user_email,
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to upload to Drive: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception uploading to Drive for %s", user_email)
        raise GoogleDriveError("Exception uploading to Drive") from exc


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
                    logger.info(
                        "Updated Drive file %s for %s",
                        file_id,
                        user_email,
                    )
                    return True
                err = await resp.text()
                logger.error(
                    "Failed to update Drive file %s for %s: %s %s",
                    file_id,
                    user_email,
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to update Drive file {file_id}: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception(
            "Exception updating Drive file %s for %s",
            file_id,
            user_email,
        )
        raise GoogleDriveError("Exception updating Drive file") from exc


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
                logger.error(
                    "Failed to download Drive file %s: %s %s",
                    file_id,
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to download Drive file {file_id}: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception downloading Drive file %s", file_id)
        raise GoogleDriveError("Exception downloading Drive file") from exc


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
                logger.error(
                    "Failed to list Drive files: %s %s",
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to list Drive files: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception listing Drive files")
        raise GoogleDriveError("Exception listing Drive files") from exc


async def get_drive_storage_quota(refresh_token: Optional[str]) -> Optional[dict]:
    """Return Google Drive storage quota information for the authenticated user."""
    access_token = await get_access_token_from_refresh(refresh_token)
    if not access_token:
        logger.warning("No Google access token available for Drive quota lookup")
        return None

    params = {
        "fields": "storageQuota(limit,usage,usageInDrive,usageInDriveTrash)",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/drive/v3/about",
                headers=headers,
                params=params,
            ) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.json()
                    quota = data.get("storageQuota")
                    if not quota:
                        logger.warning("Drive quota response missing storageQuota field")
                        return None

                    def _to_int(value):
                        try:
                            if value in (None, ""):
                                return None
                            return int(value)
                        except (TypeError, ValueError):
                            return None

                    return {
                        "limit": _to_int(quota.get("limit")),
                        "usage": _to_int(quota.get("usage")),
                        "usage_in_drive": _to_int(quota.get("usageInDrive")),
                        "usage_in_drive_trash": _to_int(quota.get("usageInDriveTrash")),
                    }
                err = await resp.text()
                logger.error(
                    "Failed to fetch Drive storage quota: %s %s",
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to fetch Drive storage quota: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception fetching Drive storage quota")
        raise GoogleDriveError("Exception fetching Drive storage quota") from exc


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
                logger.error(
                    "Failed to search Drive file %s: %s %s",
                    filename,
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to search Drive file {filename}: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception("Exception finding Drive file by name %s", filename)
        raise GoogleDriveError("Exception finding Drive file by name") from exc


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
                    logger.info(
                        "Deleted Drive file %s for %s",
                        file_id,
                        user_email or "unknown user",
                    )
                    return True
                err = await resp.text()
                logger.error(
                    "Failed to delete Drive file %s for %s: %s %s",
                    file_id,
                    user_email,
                    resp.status,
                    err,
                )
                raise GoogleDriveError(
                    f"Failed to delete Drive file {file_id}: {resp.status} {err}"
                )
    except GoogleAPIError:
        raise
    except Exception as exc:
        logger.exception(
            "Exception deleting Drive file %s for %s",
            file_id,
            user_email,
        )
        raise GoogleDriveError("Exception deleting Drive file") from exc

