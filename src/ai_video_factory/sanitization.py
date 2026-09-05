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
_SHELL_ATOM = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;&]+)'''
_QUOTED_OR_ATOM = _SHELL_ATOM
_SERIALIZED_STRING = r'''(?:b)?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
_ESCAPED_OPTION_NAME = re.compile(
    r"--(?:[a-z0-9._-]|\\u[0-9a-f]{4})+", re.IGNORECASE
)
_SERIALIZED_AUTHORIZATION_ARGUMENT = re.compile(
    rf'''(?ix)
    (?P<prefix>(?:b)?["']--authorization["']\s*,\s*)
    {_SERIALIZED_STRING}
    (?:\s*,\s*(?!(?:b)?["']--){_SERIALIZED_STRING})?
    '''
)
_SERIALIZED_COMMAND_ARGUMENT = re.compile(
    rf'''(?ix)
    (?P<prefix>(?:b)?["']--{_SENSITIVE_NAME_PATTERN}["']\s*,\s*)
    {_SERIALIZED_STRING}
    '''
)
_SERIALIZED_SENSITIVE_ASSIGNMENT = re.compile(
    rf'''(?ix)
    (?<![a-z0-9._-])
    (?P<prefix>(?:b)?["']?{_SENSITIVE_NAME_PATTERN}["']?\s*[:=]\s*)
    (?:\[[^\]\r\n]*\]|\([^\)\r\n]*\)|\{{[^\}}\r\n]*\}})
    '''
)
_AUTHORIZATION_COMMAND_ARGUMENT = re.compile(
    rf'''(?ix)
    (?P<prefix>["']?--authorization["']?(?:\s+|=))
    (?P<first>{_SHELL_ATOM})
    (?:\s+(?P<second>(?!--){_SHELL_ATOM}))?
    '''
)
_COMMAND_ARGUMENT = re.compile(
    rf"(?i)(?P<prefix>[\"']?--{_SENSITIVE_NAME_PATTERN}[\"']?(?:\s+|=))"
    rf"{_QUOTED_OR_ATOM}"
)
_AUTHORIZATION_ASSIGNMENT = re.compile(
    rf'''(?ix)
    (?<![a-z0-9._-])
    (?P<prefix>["']?authorization["']?\s*[:=]\s*)
    (?P<first>{_SHELL_ATOM})
    (?:\s+(?P<second>{_SHELL_ATOM}))?
    '''
)
_ASSIGNMENT = re.compile(
    rf"(?i)(?<![a-z0-9._-])"
    rf"(?P<prefix>[\"']?{_SENSITIVE_NAME_PATTERN}[\"']?\s*[:=]\s*)"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&\r\n]+)"
)


def _is_sensitive_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    if normalized.startswith("deauthorization"):
        return False
    return normalized in {"sig", "signature"} or normalized.endswith("signature") or any(
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


def _normalize_sensitive_option_name(match: re.Match[str]) -> str:
    raw_name = match.group(0)

    def decode_ascii_escape(escape: re.Match[str]) -> str:
        character = chr(int(escape.group(1), 16))
        return character if character.isascii() else escape.group(0)

    decoded_name = re.sub(
        r"\\u([0-9a-f]{4})", decode_ascii_escape, raw_name, flags=re.IGNORECASE
    )
    if _is_sensitive_name(decoded_name[2:]):
        return decoded_name
    return raw_name


def _looks_like_diagnostic_context(atom: str) -> bool:
    if atom[:1] in {'"', "'"}:
        return False
    return bool(
        re.fullmatch(r"[a-z_][a-z0-9_.-]*=.*", atom, re.IGNORECASE)
        or re.match(r"(?:/|\.{1,2}/)", atom)
        or re.fullmatch(r"[^/\s]+\.[a-z0-9]{1,10}", atom, re.IGNORECASE)
    )


def _looks_like_standalone_credential(atom: str) -> bool:
    unquoted = atom.strip('"\'')
    normalized = re.sub(r"[^a-z0-9]", "", unquoted.casefold())
    if normalized in {"credential", "password", "secret", "token"}:
        return False
    return any(
        marker in normalized
        for marker in ("credential", "password", "secret", "token")
    )


def _redact_authorization_value(match: re.Match[str]) -> str:
    second = match.group("second")
    preserved_context = (
        f" {second}"
        if second
        and _looks_like_standalone_credential(match.group("first"))
        and _looks_like_diagnostic_context(second)
        else ""
    )
    return f"{match.group('prefix')}[REDACTED]{preserved_context}"


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
        ).replace("%5BREDACTED%5D", "[REDACTED]")
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    except (TypeError, ValueError):
        return "[REDACTED URL]"


def sanitize_diagnostic(value: object, *, max_chars: int = MAX_DIAGNOSTIC_CHARS) -> str:
    """Redact common credential forms and cap diagnostic text to a stable size."""
    text = str(value)
    text = _ESCAPED_OPTION_NAME.sub(_normalize_sensitive_option_name, text)
    text = _URL.sub(_redact_url, text)
    text = _HEADER.sub(lambda match: f"{match.group(1)}: [REDACTED]", text)
    text = _SERIALIZED_AUTHORIZATION_ARGUMENT.sub(
        lambda match: f'{match.group("prefix")}"[REDACTED]"', text
    )
    text = _SERIALIZED_COMMAND_ARGUMENT.sub(
        lambda match: f'{match.group("prefix")}"[REDACTED]"', text
    )
    text = _SERIALIZED_SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]", text
    )
    text = _AUTHORIZATION_COMMAND_ARGUMENT.sub(
        _redact_authorization_value, text
    )
    text = _COMMAND_ARGUMENT.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]", text
    )
    text = _AUTHORIZATION_ASSIGNMENT.sub(
        _redact_authorization_value, text
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
