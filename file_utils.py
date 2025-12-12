"""File storage abstraction layer for transcription service.

This module defines the interface that both Google Drive and local filesystem
backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple


class FileStorageBackend(ABC):
    """Abstract base class for file storage backends."""
    
    @abstractmethod
    async def ensure_folder(self, user_identifier: Optional[str] = None) -> Optional[str]:
        """Ensure the storage folder exists and return its identifier."""
        pass
    
    @abstractmethod
    async def upload_file(
        self,
        filename: str,
        file_data: bytes,
        mime_type: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Upload a file to storage."""
        pass
    
    @abstractmethod
    async def update_file(
        self,
        file_id: str,
        file_data: bytes,
        mime_type: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Update an existing file in storage."""
        pass
    
    @abstractmethod
    async def download_file_bytes(
        self,
        file_id: str,
        user_identifier: Optional[str] = None
    ) -> Optional[bytes]:
        """Download a file as raw bytes."""
        pass
    
    @abstractmethod
    async def find_file_by_name(
        self,
        filename: str,
        user_identifier: Optional[str] = None
    ) -> Optional[str]:
        """Find a file by name and return its identifier."""
        pass
    
    @abstractmethod
    async def delete_file(
        self,
        file_id: str,
        user_identifier: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """Delete a file from storage."""
        pass
    
    @abstractmethod
    async def get_file_metadata(
        self,
        file_id: str,
        user_identifier: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a file, including size."""
        pass
    
    @abstractmethod
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
        pass

