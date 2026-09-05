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


@pytest.mark.parametrize(
    ("raw", "secrets", "visible"),
    [
        ('--token "top secret value" --mode safe', ("top", "secret value"), "--mode safe"),
        ("--private-key='multi word key' --mode safe", ("multi word key",), "--mode safe"),
        ("--authorization Bearer bearer-secret --mode safe", ("bearer-secret",), "--mode safe"),
        ("AUTHORIZATION=Bearer assignment-secret", ("assignment-secret",), "AUTHORIZATION="),
        ("Command ['--password', 'list-secret'] failed", ("list-secret",), "Command"),
        ('Command ["--password", "json secret value", "--mode", "safe"] failed',
         ("json secret value",), '"--mode", "safe"'),
    ],
)
def test_sanitize_diagnostic_redacts_compound_credential_forms(
    raw: str, secrets: tuple[str, ...], visible: str
) -> None:
    sanitized = sanitize_diagnostic(raw)

    for secret in secrets:
        assert secret not in sanitized
    assert "[REDACTED]" in sanitized
    assert visible in sanitized


@pytest.mark.parametrize(
    "query_name",
    ["Signature", "sig", "X-Amz-Signature", "X-Goog-Signature"],
)
def test_sanitize_diagnostic_redacts_signed_url_query_values(query_name: str) -> None:
    raw = f"https://example.test/object?{query_name}=signed-secret&safe=value"

    sanitized = sanitize_diagnostic(raw)

    assert "signed-secret" not in sanitized
    assert "[REDACTED]" in sanitized
    assert "safe=value" in sanitized


def test_sanitize_diagnostic_preserves_non_sensitive_diagnostics() -> None:
    raw = "command --mode safe assignment=normal https://example.test/?safe=value"

    assert sanitize_diagnostic(raw) == raw


@pytest.mark.parametrize(
    ("raw", "secrets", "visible"),
    [
        (
            "--authorization Token token-secret --mode safe",
            ("Token", "token-secret"),
            "--mode safe",
        ),
        (
            "AUTHORIZATION=Token assignment-secret status=failed",
            ("Token", "assignment-secret"),
            "status=failed",
        ),
        (
            "Command ['--authorization', 'Token', 'list-token-secret', '--mode', 'safe'] failed",
            ("Token", "list-token-secret"),
            "'--mode', 'safe'",
        ),
        (
            'Command "--password" "quoted-flag-secret" --mode safe failed',
            ("quoted-flag-secret",),
            "--mode safe",
        ),
        (
            "Command [b'--password', b'bytes-secret'] failed",
            ("bytes-secret",),
            "Command",
        ),
        (
            r'["--pass\u0077ord", "unicode-escaped-secret", "--mode", "safe"]',
            ("unicode-escaped-secret",),
            '"--mode", "safe"',
        ),
        (
            '--authorization Bearer "multi word token" --mode safe',
            ("multi word token",),
            "--mode safe",
        ),
        (
            'AUTHORIZATION=Bearer "multi word token" status=failed',
            ("multi word token",),
            "status=failed",
        ),
        (
            '--authorization "Bearer" "separate-token" --mode safe',
            ("Bearer", "separate-token"),
            "--mode safe",
        ),
        (
            "--authorization Custom secret.txt --mode safe",
            ("Custom", "secret.txt"),
            "--mode safe",
        ),
        (
            "--authorization Custom /token-value --mode safe",
            ("Custom", "/token-value"),
            "--mode safe",
        ),
        (
            "AUTHORIZATION=Custom secret.txt status=failed",
            ("Custom", "secret.txt"),
            "status=failed",
        ),
        (
            "--authorization Token secret.txt --mode safe",
            ("Token", "secret.txt"),
            "--mode safe",
        ),
        (
            "--authorization Token /credential-value --mode safe",
            ("Token", "/credential-value"),
            "--mode safe",
        ),
        (
            "AUTHORIZATION=Token secret.txt status=failed",
            ("Token", "secret.txt"),
            "status=failed",
        ),
    ],
)
def test_sanitize_diagnostic_blocks_alternate_credential_encodings(
    raw: str, secrets: tuple[str, ...], visible: str
) -> None:
    sanitized = sanitize_diagnostic(raw)

    for secret in secrets:
        assert secret not in sanitized
    assert "[REDACTED]" in sanitized
    assert visible in sanitized


@pytest.mark.parametrize(
    "raw",
    [
        "deauthorization=normal --mode safe",
        "https://example.test/?signature_algorithm=RSA-SHA256&safe=value",
        "https://example.test/?deauthorization=normal&safe=value",
    ],
)
def test_sanitize_diagnostic_preserves_similar_non_sensitive_names(raw: str) -> None:
    assert sanitize_diagnostic(raw) == raw


@pytest.mark.parametrize(
    ("raw", "visible"),
    [
        ("authorization=opaque-token status=failed path=/tmp/input.mp4", "status=failed"),
        ("--authorization opaque-token input.mp4 --mode safe", "input.mp4"),
    ],
)
def test_sanitize_diagnostic_preserves_context_after_single_authorization_value(
    raw: str, visible: str
) -> None:
    sanitized = sanitize_diagnostic(raw)

    assert "opaque-token" not in sanitized
    assert visible in sanitized


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ('{"authorization": ["Bearer", "leaked-json-secret"]}', "leaked-json-secret"),
        ("{'authorization': ['Bearer', 'leaked-python-secret']}", "leaked-python-secret"),
        ("{b'authorization': [b'Bearer', b'leaked-bytes-secret']}", "leaked-bytes-secret"),
        ("{'authorization': ('Bearer', 'python-tuple-secret')}", "python-tuple-secret"),
        ("{b'authorization': (b'Bearer', b'bytes-tuple-secret')}", "bytes-tuple-secret"),
    ],
)
def test_sanitize_diagnostic_redacts_serialized_sensitive_mapping_containers(
    raw: str, secret: str
) -> None:
    sanitized = sanitize_diagnostic(raw)

    assert secret not in sanitized
    assert "[REDACTED]" in sanitized
