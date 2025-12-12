"""Local filesystem backend for file storage."""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from file_utils import FileStorageBackend

logger = logging.getLogger("transcribe_service.local_file_utils")

# Base directory for local storage
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", "local_data")


class LocalFileStorageBackend(FileStorageBackend):
    """Local filesystem implementation of file storage backend."""
    
    def __init__(self, base_dir: str = LOCAL_DATA_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_user_dir(self, user_identifier: Optional[str] = None) -> Path:
        """Get the directory for a specific user."""
        if user_identifier:
            user_dir = self.base_dir / user_identifier
            user_dir.mkdir(parents=True, exist_ok=True)
            return user_dir
        return self.base_dir
    
    async def ensure_folder(self, user_identifier: Optional[str] = None) -> Optional[str]:
        """Ensure the storage folder exists and return its identifier."""
        user_dir = self._get_user_dir(user_identifier)
        return str(user_dir)
    
    async def upload_file(
        self,
        filename: str,
        file_data: bytes,
        mime_type: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Upload a file to local storage."""
        try:
            user_dir = self._get_user_dir(user_identifier)
            file_path = user_dir / filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(
                "Uploaded file to local storage for %s as %s",
                user_email or "unknown user",
                filename,
            )
            return True
        except Exception as exc:
            logger.exception("Exception uploading to local storage for %s", user_email)
            return False
    
    async def update_file(
        self,
        file_id: str,
        file_data: bytes,
        mime_type: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Update an existing file in local storage."""
        try:
            # file_id is the full path in local mode
            file_path = Path(file_id)
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(
                "Updated local file %s for %s",
                file_id,
                user_email or "unknown user",
            )
            return True
        except Exception as exc:
            logger.exception(
                "Exception updating local file %s for %s",
                file_id,
                user_email,
            )
            return False
    
    async def download_file_bytes(
        self,
        file_id: str,
        user_identifier: Optional[str] = None
    ) -> Optional[bytes]:
        """Download a file as raw bytes."""
        try:
            # file_id is the full path in local mode
            file_path = Path(file_id)
            
            if not file_path.exists():
                logger.warning("File not found: %s", file_id)
                return None
            
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as exc:
            logger.exception("Exception downloading local file %s", file_id)
            return None
    
    async def find_file_by_name(
        self,
        filename: str,
        user_identifier: Optional[str] = None
    ) -> Optional[str]:
        """Find a file by name and return its full path."""
        try:
            user_dir = self._get_user_dir(user_identifier)
            file_path = user_dir / filename
            
            if file_path.exists():
                return str(file_path)
            return None
        except Exception as exc:
            logger.exception("Exception finding local file by name %s", filename)
            return None
    
    async def delete_file(
        self,
        file_id: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Delete a file from local storage."""
        try:
            # file_id is the full path in local mode
            file_path = Path(file_id)
            
            if file_path.exists():
                file_path.unlink()
                logger.info(
                    "Deleted local file %s for %s",
                    file_id,
                    user_email or "unknown user",
                )
                return True
            return False
        except Exception as exc:
            logger.exception(
                "Exception deleting local file %s for %s",
                file_id,
                user_email,
            )
            return False
    
    async def get_file_metadata(
        self,
        file_id: str,
        user_identifier: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a file, including size."""
        try:
            # file_id is the full path in local mode
            file_path = Path(file_id)
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            return {
                "id": str(file_path),
                "name": file_path.name,
                "size": str(stat.st_size),
                "mimeType": "application/octet-stream",  # Default for local files
            }
        except Exception as exc:
            logger.exception("Exception getting local file metadata %s", file_id)
            return None
    
    async def stream_file_range(
        self,
        file_id: str,
        range_header: Optional[str] = None,
        user_identifier: Optional[str] = None
    ) -> Optional[Tuple[bytes, int, int, int, int]]:
        """
        Stream a file with optional range support.
        Returns tuple of (content, status_code, start_byte, end_byte, total_size) or None.
        """
        try:
            # file_id is the full path in local mode
            file_path = Path(file_id)
            
            if not file_path.exists():
                return None
            
            total_size = file_path.stat().st_size
            
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
            
            # Read the requested range
            with open(file_path, 'rb') as f:
                f.seek(start_byte)
                content = f.read(end_byte - start_byte + 1)
            
            status_code = 206 if range_header else 200
            return (content, status_code, start_byte, end_byte, total_size)
        except Exception as exc:
            logger.exception("Exception streaming local file %s", file_id)
            return None

