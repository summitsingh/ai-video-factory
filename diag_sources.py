"""Test a batch of NASA search endpoints for extraction success + claim text, so the
pilot can select OVERLAPPING sources whose claims merge into corroborated (>=2 source)
verified facts. Uses max_attempts=2 (current production setting). Reports per-source:
success/fail and up to 5 normalized claims with their verbatim quotes."""

import sys
import time

from ai_video_factory.research_pipeline import (
    SourceContent,
    _clean_excerpt,
    make_production_extractor,
    make_secure_http_transport,
)

TOPIC = "Water Beyond Earth: Where It Is and Why It Means We are Looking"
QUERIES = [
    ("Mars+ancient+riverbed", "Mars ancient rivers"),
    ("Mars+ancient+oceans", "Mars ancient oceans"),
    ("Mars+polar+ice+caps", "Mars polar ice"),
    ("Mars+subsurface+water", "Mars subsurface water"),
    ("Enceladus+plume", "Enceladus plume"),
    ("Enceladus+subsurface+ocean", "Enceladus ocean"),
    ("Europa+subsurface+ocean", "Europa ocean"),
    ("Lunar+polar+ice", "Lunar polar ice"),
]


def main() -> int:
    transport = make_secure_http_transport()
    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:1234/v1/chat/completions", max_attempts=2
    )

    def norm(s: str) -> str:
        return " ".join((s or "").strip().lower().rstrip(".").split())

    for url, label in QUERIES:
        full = f"https://images-api.nasa.gov/search?q={url}"
        s = time.monotonic()
        try:
            raw = transport(full)
            text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else (raw or "")
            cleaned = _clean_excerpt(text, TOPIC).text
            sc = SourceContent(url=full, title="", quality="high", content=cleaned, content_hash="")
            outs = extractor(TOPIC, [sc])
            ok = sum(1 for e in outs if (getattr(e, "quote", "") or "").strip() in cleaned)
            dt = time.monotonic() - s
            print(f"\n[OK ] {label:24s} {dt:5.1f}s -> {ok}/{len(outs)} quoted-in-excerpt", file=sys.stderr)
            for e in outs[:5]:
                print(f"      claim: {norm(e.claim)[:95]}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            dt = time.monotonic() - s
            print(f"\n[FAIL] {label:24s} {dt:5.1f}s -> {type(exc).__name__}: {str(exc)[:55]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
