"""Tests for source verification and placeholder guard (item 4)."""

import pytest

from ai_video_factory import source_verifier as sv


def test_guard_rejects_placeholder_urls():
    placeholders = [
        "https://example.com/fermi",
        "http://example.org",
        "https://mysite.placeholder/research",
        "https://lorem-ipsum.com/article",
        "https://news.example.net/story",
        "https://example.com",  # the nightly-batch fallback URL
    ]
    for url in placeholders:
        with pytest.raises(
            sv.SourceVerificationError, match="placeholder source URL"
        ):
            sv.guard_no_placeholders([url])


def test_guard_accepts_real_urls():
    real = [
        "https://en.wikipedia.org/wiki/Fermi_paradox",
        "https://www.seti.org/research/seti-101/fermi-paradox/",
        "https://astrobiology.nasa.gov/",
        "https://arxiv.org/abs/2509.22878v2",
    ]
    sv.guard_no_placeholders(real)  # must not raise
    sv.guard_no_placeholders([])  # empty is fine
    sv.guard_no_placeholders(None)  # none is fine


def test_seed_fermi_sources():
    seeds = sv.seed_fermi_sources()
    assert len(seeds) == 5
    assert all(url.startswith("https://") for url in seeds)
    assert any("wikipedia.org/wiki/Fermi_paradox" in url for url in seeds)
    assert any("seti.org" in url for url in seeds)
    assert any("nasa.gov" in url for url in seeds)
    assert sum("arxiv.org" in url for url in seeds) >= 2
    # Seeds themselves must never trip the placeholder guard.
    sv.guard_no_placeholders(seeds)


def test_is_fermi_topic():
    assert sv.is_fermi_topic("The Fermi Paradox")
    assert sv.is_fermi_topic("fermi paradox: where is everybody")
    assert not sv.is_fermi_topic("Water on Mars")
    assert not sv.is_fermi_topic(None)


def test_verify_sources_splits_live_and_dead(monkeypatch):
    statuses = {
        "https://live.example.org/a": True,
        "https://dead.example.org/b": False,
    }
    monkeypatch.setattr(
        sv, "verify_url", lambda url, timeout=15.0: statuses[url]
    )
    report = sv.verify_sources(
        ["https://live.example.org/a", "https://dead.example.org/b",
         "https://live.example.org/a"]  # duplicate checked once
    )
    assert report.live == ["https://live.example.org/a"]
    assert report.dead == ["https://dead.example.org/b"]
    assert report.checked_at


def test_verify_url_head_then_get_fallback(monkeypatch):
    calls = []

    def fake_status(url, method, timeout):
        calls.append(method)
        if method == "HEAD":
            return 405  # server rejects HEAD
        return 200

    monkeypatch.setattr(sv, "_request_status", fake_status)
    assert sv.verify_url("https://some-site.test/page") is True
    assert calls == ["HEAD", "GET"]


def test_verify_url_rejects_non_200_and_non_http(monkeypatch):
    monkeypatch.setattr(sv, "_request_status", lambda url, method, timeout: 404)
    assert sv.verify_url("https://gone.test/page") is False
    assert sv.verify_url("not a url") is False
    assert sv.verify_url("") is False
