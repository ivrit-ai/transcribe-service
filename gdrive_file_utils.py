"""Google Drive backend implementing FileStorageBackend interface."""

import os
import json
import hashlib
import logging
import aiohttp
import dotenv
from typing import Optional, Dict, Any, Tuple
from file_utils import FileStorageBackend
from gdrive_auth import (
    get_access_token_from_refresh,
    GoogleDriveError,
)

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger("transcribe_service.gdrive_file_utils")

DRIVE_FOLDER_NAME = os.environ.get("GOOGLE_DRIVE_FOLDER_NAME", "transcribe.ivrit.ai")
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

# Cache folder IDs per refresh token to avoid repeated Drive lookups
folder_id_cache: Dict[str, str] = {}


def _get_folder_cache_key(refresh_token: str) -> str:
    """Generate a cache key from a refresh token."""
    return hashlib.sha256(refresh_token.encode()).hexdigest()


class GoogleDriveStorageBackend(FileStorageBackend):
    """Google Drive implementation of file storage backend."""
    
    def __init__(self):
        pass
    
    async def ensure_folder(self, user_identifier: Optional[str] = None) -> Optional[str]:
        """Ensure the application Drive folder exists and return its ID."""
        if not user_identifier:
            return None

        cache_key = _get_folder_cache_key(user_identifier)
        cached = folder_id_cache.get(cache_key)
        if cached:
            return cached

        token = await get_access_token_from_refresh(user_identifier)
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
        except GoogleDriveError:
            raise
        except Exception as exc:
            logger.exception("Exception ensuring Drive folder")
            raise GoogleDriveError("Exception ensuring Drive folder") from exc
    
    async def upload_file(
        self,
        filename: str,
        file_data: bytes,
        mime_type: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Upload a file into the application's Drive folder."""
        access_token = await get_access_token_from_refresh(user_identifier)
        if not access_token:
            logger.warning(f"No Google access token available; skipping Drive upload for {user_email or 'unknown user'}")
            return False

        folder_id = await self.ensure_folder(user_identifier)
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
        except GoogleDriveError:
            raise
        except Exception as exc:
            logger.exception("Exception uploading to Drive for %s", user_email)
            raise GoogleDriveError("Exception uploading to Drive") from exc
    
    async def update_file(
        self,
        file_id: str,
        file_data: bytes,
        mime_type: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Update an existing file in the application's Drive folder."""
        access_token = await get_access_token_from_refresh(user_identifier)
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
        except GoogleDriveError:
            raise
        except Exception as exc:
            logger.exception(
                "Exception updating Drive file %s for %s",
                file_id,
                user_email,
            )
            raise GoogleDriveError("Exception updating Drive file") from exc
    
    async def download_file_bytes(
        self,
        file_id: str,
        user_identifier: Optional[str] = None
    ) -> Optional[bytes]:
        """Download a specific file from the application's Drive folder as raw bytes."""
        access_token = await get_access_token_from_refresh(user_identifier)
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
        except GoogleDriveError:
            raise
        except Exception as exc:
            logger.exception("Exception downloading Drive file %s", file_id)
            raise GoogleDriveError("Exception downloading Drive file") from exc
    
    async def find_file_by_name(
        self,
        filename: str,
        user_identifier: Optional[str] = None
    ) -> Optional[str]:
        """Find a file by name within the application's Drive folder and return its ID."""
        access_token = await get_access_token_from_refresh(user_identifier)
        if not access_token:
            return None

        folder_id = await self.ensure_folder(user_identifier)
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
        except GoogleDriveError:
            raise
        except Exception as exc:
            logger.exception("Exception finding Drive file by name %s", filename)
            raise GoogleDriveError("Exception finding Drive file by name") from exc
    
    async def delete_file(
        self,
        file_id: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Delete a file from Google Drive by its file ID."""
        access_token = await get_access_token_from_refresh(user_identifier)
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
        except GoogleDriveError:
            raise
        except Exception as exc:
            logger.exception(
                "Exception deleting Drive file %s for %s",
                file_id,
                user_email,
            )
            raise GoogleDriveError("Exception deleting Drive file") from exc
    
    async def get_file_metadata(
        self,
        file_id: str,
        user_identifier: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a file in Google Drive, including size."""
        access_token = await get_access_token_from_refresh(user_identifier)
        if not access_token:
            logger.warning("No Google access token available for getting file metadata")
            return None
        
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}",
                    params={"fields": "id,name,size,mimeType"},
                    headers=headers,
                ) as resp:
                    if 200 <= resp.status < 300:
                        return await resp.json()
                    err = await resp.text()
                    logger.error(
                        "Failed to get Drive file metadata %s: %s %s",
                        file_id,
                        resp.status,
                        err,
                    )
                    return None
        except Exception as exc:
            logger.exception("Exception getting Drive file metadata %s", file_id)
            return None
    
    async def stream_file_range(
        self,
        file_id: str,
        range_header: Optional[str] = None,
        user_identifier: Optional[str] = None
    ) -> Optional[Tuple[bytes, int, int, int, int]]:
        """
        Stream a file from Google Drive with optional range support.
        Returns tuple of (content, status_code, start_byte, end_byte, total_size) or None.
        """
        access_token = await get_access_token_from_refresh(user_identifier)
        if not access_token:
            logger.warning("No Google access token available for streaming Drive file")
            return None
        
        # First get file metadata to know the total size
        metadata = await self.get_file_metadata(file_id, user_identifier)
        if not metadata or "size" not in metadata:
            logger.error(f"Could not get size for file {file_id}")
            return None
        
        total_size = int(metadata["size"])
        
        # Parse range header if provided
        start_byte = 0
        end_byte = total_size - 1
        
        if range_header:
            # Format: "bytes=start-end" or "bytes=start-"
            try:
                range_str = range_header.replace("bytes=", "").strip()
                if "-" in range_str:
                    parts = range_str.split("-")
                    if parts[0]:
                        start_byte = int(parts[0])
                    if parts[1]:
                        end_byte = int(parts[1])
                    else:
                        end_byte = total_size - 1
                        
                    # Validate range
                    if start_byte >= total_size:
                        start_byte = 0
                    if end_byte >= total_size:
                        end_byte = total_size - 1
            except ValueError:
                logger.warning(f"Invalid range header: {range_header}")
                start_byte = 0
                end_byte = total_size - 1
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Range": f"bytes={start_byte}-{end_byte}",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}",
                    params={"alt": "media"},
                    headers=headers,
                ) as resp:
                    if resp.status in (200, 206):
                        content = await resp.read()
                        return (content, resp.status, start_byte, end_byte, total_size)
                    err = await resp.text()
                    logger.error(
                        "Failed to stream Drive file %s: %s %s",
                        file_id,
                        resp.status,
                        err,
                    )
                    return None
        except Exception as exc:
            logger.exception("Exception streaming Drive file %s", file_id)
            return None
