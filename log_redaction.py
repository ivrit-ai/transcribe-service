"""Redaction of secrets from log output.

Under xhost the runtime log is read from outside the container, so every line
the process emits is retrievable by anyone with access to the hosting platform.
Redaction happens in the formatter, the last step before a record becomes text,
so it covers exception tracebacks as well as the message itself.
"""

import logging
import os
import re
from urllib.parse import urlsplit

from loggingredactor import RedactingFilter

MASK = "<redacted>"

_SECRET_ENV_NAME_RE = re.compile(r"SECRET|TOKEN|KEY|PASSWORD|PASSWD", re.IGNORECASE)
_MIN_SECRET_LENGTH = 8

# Credentials that reach us from outside the environment: Google access tokens,
# refresh tokens and authorization codes, RunPod keys users supply themselves,
# and anything spelled out as a labelled secret in a URL, header or JSON body.
_PATTERNS = [
    re.compile(r"ya29\.[A-Za-z0-9._\-]{10,}"),
    re.compile(r"1//[A-Za-z0-9._\-]{10,}"),
    re.compile(r"\b4/0[A-Za-z0-9._\-]{10,}"),
    re.compile(r"\brpa_[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)(?<=\bbearer )[A-Za-z0-9._\-]+"),
]

# Keeps the label, drops the value; a lookbehind can't express this because the
# separator is variable width, so it goes in as a callable redactor.
_LABELLED_SECRET_RE = re.compile(
    r"(?i)\b(access_token|refresh_token|id_token|client_secret|api_key|apikey|password)"
    r"(['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}&]+"
)


def _redact_labelled_secrets(text, mask):
    return _LABELLED_SECRET_RE.sub(lambda m: m.group(1) + m.group(2) + mask, text)


def _environment_secrets():
    """Exact secret values visible in the environment, longest first."""
    values = set()
    for name, value in os.environ.items():
        if _SECRET_ENV_NAME_RE.search(name) and len(value) >= _MIN_SECRET_LENGTH:
            values.add(value)

    password = urlsplit(os.environ.get("DATABASE_URL", "")).password
    if password and len(password) >= _MIN_SECRET_LENGTH:
        values.add(password)

    return sorted(values, key=len, reverse=True)


def _build_redactor():
    patterns = [re.compile(re.escape(secret)) for secret in _environment_secrets()]
    patterns.extend(_PATTERNS)
    patterns.append(_redact_labelled_secrets)
    return RedactingFilter(mask_patterns=patterns, mask=MASK)


_redactor = _build_redactor()


class RedactingFormatter(logging.Formatter):
    """Delegates formatting to another formatter, then redacts the result."""

    def __init__(self, inner: logging.Formatter):
        super().__init__()
        self.inner = inner

    def format(self, record: logging.LogRecord) -> str:
        return _redactor.redact(self.inner.format(record))


def redact_handlers(logger: logging.Logger) -> None:
    """Wrap the formatters of a logger's own handlers (e.g. uvicorn's)."""
    for handler in logger.handlers:
        if isinstance(handler.formatter, RedactingFormatter):
            continue
        handler.setFormatter(RedactingFormatter(handler.formatter or logging.Formatter()))
