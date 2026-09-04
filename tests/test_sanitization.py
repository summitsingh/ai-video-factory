import pytest

from ai_video_factory.sanitization import MAX_DIAGNOSTIC_CHARS, sanitize_diagnostic


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("TOKEN=token-value", "token-value"),
        ("api_key: key-value", "key-value"),
        ('{"password": "password-value"}', "password-value"),
        ("--private-key private-key-value", "private-key-value"),
        ("Authorization: Bearer authorization-value", "authorization-value"),
        ("Cookie: session=cookie-value", "cookie-value"),
        ("Set-Cookie: session=set-cookie-value", "set-cookie-value"),
    ],
)
def test_sanitize_diagnostic_redacts_credentials(raw: str, secret: str) -> None:
    """Removing a credential redaction would expose the named secret."""
    sanitized = sanitize_diagnostic(raw)

    assert secret not in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_diagnostic_redacts_url_credentials_and_sensitive_query_values() -> None:
    """Credential-bearing URLs must be safe in both public and persisted errors."""
    raw = "https://alice:basic-secret@example.test/video?token=query-secret&safe=value"

    sanitized = sanitize_diagnostic(raw)

    assert "alice" not in sanitized
    assert "basic-secret" not in sanitized
    assert "query-secret" not in sanitized
    assert "safe=value" in sanitized


def test_sanitize_diagnostic_limits_oversized_output_after_redaction() -> None:
    """Unbounded subprocess output must not inflate manifests or JSON responses."""
    raw = "password=oversized-secret\n" + ("x" * (MAX_DIAGNOSTIC_CHARS * 2))

    sanitized = sanitize_diagnostic(raw)

    assert "oversized-secret" not in sanitized
    assert len(sanitized) <= MAX_DIAGNOSTIC_CHARS
    assert sanitized.endswith("[truncated]")
