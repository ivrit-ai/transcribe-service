"""Encrypted client-side session cookie and OAuth state cookie.

The server keeps no session state: everything a session needs rides in a
Fernet-encrypted cookie, so restarts and deploys do not sign anyone out. The
payload is encrypted rather than just signed because it carries the user's
Google refresh token and RunPod key.
"""

import json
import logging
import time
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from fastapi import Request, Response

SESSION_PAYLOAD_VERSION = 1
SESSION_MAX_AGE_SECONDS = 30 * 24 * 3600
SESSION_RENEW_AFTER_SECONDS = 24 * 3600
OAUTH_STATE_MAX_AGE_SECONDS = 600

logger = logging.getLogger("transcribe_service.session_cookie")


class AuthCookies:
    """Reads and writes the session cookie and the OAuth state cookie."""

    def __init__(self, encryption_key: Optional[str], secure: bool):
        if not encryption_key:
            raise RuntimeError("SESSION_ENCRYPTION_KEY is required outside local mode")
        try:
            self.fernet = Fernet(encryption_key)
        except ValueError:
            # Not chained, so nothing derived from the key reaches the startup traceback.
            raise RuntimeError("SESSION_ENCRYPTION_KEY is not a valid Fernet key") from None
        self.secure = secure
        # __Host- requires Secure; without TLS (dev) the browser would drop it.
        self.session_cookie_name = "__Host-session" if secure else "session"
        self.oauth_state_cookie_name = "__Host-oauth_state" if secure else "oauth_state"

    def read_session(self, request: Request) -> Optional[dict]:
        """Return the session payload, or None if missing, tampered, expired or of another version."""
        if not hasattr(request.state, "session"):
            request.state.session = self._decode_session(request)
        return request.state.session

    def _decode_session(self, request: Request) -> Optional[dict]:
        token = request.cookies.get(self.session_cookie_name)
        if not token:
            return None
        try:
            session = json.loads(self.fernet.decrypt(token, ttl=SESSION_MAX_AGE_SECONDS))
        except (InvalidToken, ValueError):
            logger.debug("Ignoring an invalid or expired session cookie")
            return None
        if not isinstance(session, dict) or session.get("v") != SESSION_PAYLOAD_VERSION:
            logger.debug("Ignoring a session cookie with an unsupported payload")
            return None
        return session

    def session_due_for_renewal(self, request: Request) -> bool:
        """Whether the session cookie, already read successfully, is old enough to re-issue."""
        token = request.cookies[self.session_cookie_name]
        return time.time() - self.fernet.extract_timestamp(token) >= SESSION_RENEW_AFTER_SECONDS

    def write_session(self, response: Response, session: dict) -> None:
        payload = json.dumps({**session, "v": SESSION_PAYLOAD_VERSION}).encode()
        self._set_cookie(
            response,
            self.session_cookie_name,
            self.fernet.encrypt(payload).decode(),
            SESSION_MAX_AGE_SECONDS,
        )

    def clear_session(self, response: Response) -> None:
        self._delete_cookie(response, self.session_cookie_name)

    def write_oauth_state(self, response: Response, state: str) -> None:
        self._set_cookie(response, self.oauth_state_cookie_name, state, OAUTH_STATE_MAX_AGE_SECONDS)

    def read_oauth_state(self, request: Request) -> Optional[str]:
        return request.cookies.get(self.oauth_state_cookie_name)

    def clear_oauth_state(self, response: Response) -> None:
        self._delete_cookie(response, self.oauth_state_cookie_name)

    def _set_cookie(self, response: Response, name: str, value: str, max_age: int) -> None:
        response.set_cookie(
            key=name,
            value=value,
            max_age=max_age,
            path="/",
            secure=self.secure,
            httponly=True,
            samesite="lax",
        )

    def _delete_cookie(self, response: Response, name: str) -> None:
        response.delete_cookie(key=name, path="/", secure=self.secure, httponly=True, samesite="lax")
