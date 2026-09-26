"""Encrypted storage of users' RunPod API keys in their Google Drive app folder.

The key is encrypted with a server-held Fernet key and written next to the
user's TOC, so neither Drive alone nor the server alone can recover it.
"""

from typing import Optional

from cryptography.fernet import Fernet

from gdrive_auth import GoogleDriveError
from gdrive_file_utils import GoogleDriveStorageBackend

RUNPOD_KEY_FILENAME = "runpod_key.enc"
RUNPOD_KEY_MIME_TYPE = "application/octet-stream"


class RunpodKeyStore:
    """Loads, saves and deletes a user's encrypted RunPod key on Drive."""

    def __init__(self, encryption_key: Optional[str], backend: GoogleDriveStorageBackend):
        if not encryption_key:
            raise RuntimeError("RUNPOD_KEY_ENCRYPTION_KEY is required outside local mode")
        try:
            self.fernet = Fernet(encryption_key)
        except ValueError:
            # Not chained, so nothing derived from the key reaches the startup traceback.
            raise RuntimeError("RUNPOD_KEY_ENCRYPTION_KEY is not a valid Fernet key") from None
        self.backend = backend

    async def load(self, refresh_token: str) -> str:
        """Return the user's stored key, or "" if none is stored.

        Drive failures (GoogleAPIError) and undecryptable content (InvalidToken)
        propagate to the caller.
        """
        file_id = await self.backend.find_file_by_name(RUNPOD_KEY_FILENAME, refresh_token)
        if not file_id:
            return ""
        data = await self.backend.download_file_bytes(file_id, refresh_token)
        if data is None:
            return ""
        return self.fernet.decrypt(data).decode()

    async def save(self, refresh_token: str, runpod_token: str, user_email: str) -> None:
        """Encrypt and store the key, replacing any existing one."""
        data = self.fernet.encrypt(runpod_token.encode())
        file_id = await self.backend.find_file_by_name(RUNPOD_KEY_FILENAME, refresh_token)
        if file_id:
            success = await self.backend.update_file(file_id, data, RUNPOD_KEY_MIME_TYPE, refresh_token, user_email)
        else:
            success = await self.backend.upload_file(
                RUNPOD_KEY_FILENAME, data, RUNPOD_KEY_MIME_TYPE, refresh_token, user_email
            )
        if not success:
            raise GoogleDriveError("Could not write the RunPod key file to Drive")

    async def delete(self, refresh_token: str, user_email: str) -> None:
        """Delete the stored key, including duplicates left by concurrent saves."""
        while file_id := await self.backend.find_file_by_name(RUNPOD_KEY_FILENAME, refresh_token):
            if not await self.backend.delete_file(file_id, refresh_token, user_email):
                raise GoogleDriveError("Could not delete the RunPod key file from Drive")
