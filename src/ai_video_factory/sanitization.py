"""Shared sanitization for diagnostics that may cross a persistence boundary."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MAX_DIAGNOSTIC_CHARS = 2_048
_TRUNCATION_SUFFIX = "... [truncated]"
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", re.IGNORECASE)
_HEADER = re.compile(
    r"(?im)\b(authorization|proxy-authorization|cookie|set-cookie)\s*:\s*[^\r\n]*"
)
_SENSITIVE_NAME_PATTERN = (
    r"(?:[a-z0-9]+[._-])*"
    r"(?:api[._-]?key|private[._-]?key|access[._-]?key|client[._-]?secret|"
    r"authorization|cookie|credential|password|passwd|secret|token|key)"
    r"(?:[._-][a-z0-9]+)*"
)
_COMMAND_ARGUMENT = re.compile(
    rf"(?i)(?P<prefix>--{_SENSITIVE_NAME_PATTERN}(?:\s+|=))(?P<value>[^\s,;]+)"
)
_ASSIGNMENT = re.compile(
    rf"(?i)(?P<prefix>[\"']?{_SENSITIVE_NAME_PATTERN}[\"']?\s*[:=]\s*)"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&\r\n]+)"
)


def _is_sensitive_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    return any(
        marker in normalized
        for marker in (
            "apikey",
            "privatekey",
            "accesskey",
            "clientsecret",
            "authorization",
            "cookie",
            "credential",
            "password",
            "passwd",
            "secret",
            "token",
        )
    ) or normalized == "key"


def _redact_url(match: re.Match[str]) -> str:
    raw_url = match.group(0)
    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname
        if hostname is None:
            return raw_url
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        netloc = f"[REDACTED]@{host}" if parsed.username is not None else parsed.netloc
        query = urlencode(
            [
                (key, "[REDACTED]" if _is_sensitive_name(key) else value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    except (TypeError, ValueError):
        return "[REDACTED URL]"


def sanitize_diagnostic(value: object, *, max_chars: int = MAX_DIAGNOSTIC_CHARS) -> str:
    """Redact common credential forms and cap diagnostic text to a stable size."""
    text = str(value)
    text = _URL.sub(_redact_url, text)
    text = _HEADER.sub(lambda match: f"{match.group(1)}: [REDACTED]", text)
    text = _COMMAND_ARGUMENT.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]", text
    )
    text = _ASSIGNMENT.sub(lambda match: f"{match.group('prefix')}[REDACTED]", text)

    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:max_chars]
    return text[: max_chars - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def first_diagnostic_line(value: object, *, max_chars: int = 512) -> str | None:
    """Return the first non-empty sanitized line of diagnostic output."""
    for line in sanitize_diagnostic(value, max_chars=MAX_DIAGNOSTIC_CHARS).splitlines():
        if line.strip():
            return sanitize_diagnostic(line.strip(), max_chars=max_chars)
    return None
