"""Per-source extraction diagnostic using the PILOT's exact 3 sources + config
(timeout=45, max_attempts=2). Shows timing + valid-claim count per source to determine
whether research yields >=3 grounded claims (Option B floor) or a specific source is
hanging/failing. Mirrors pilot.py build_config extractor/transport wiring."""

import time

from ai_video_factory.research_pipeline import (
    SourceContent, _clean_excerpt, make_production_extractor, make_secure_http_transport,
)

TOPIC = "Water Beyond Earth: Where It Is and Why It Means We are Looking"
SOURCES = [
    "https://images-api.nasa.gov/search?q=Mars+ancient+riverbed",
    "https://images-api.nasa.gov/search?q=Enceladus+plume",
    "https://images-api.nasa.gov/search?q=Mars+water+ice+polar",
]


def main() -> int:
    t0 = time.monotonic()
    ex = make_production_extractor(
        endpoint_url="http://127.0.0.1:1234/v1/chat/completions",
        timeout_seconds=45, max_attempts=2,
    )
    t = make_secure_http_transport()
    total_valid = 0
    for url in SOURCES:
        s = time.monotonic()
        try:
            raw = t(url)
            text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else (raw or "")
            c = _clean_excerpt(text, TOPIC).text
            sc = SourceContent(url=url, title="", quality="high", content=c, content_hash="")
            outs = ex(TOPIC, [sc])
            dt = time.monotonic() - s
            ok = sum(1 for e in outs if (getattr(e, "quote", "") or "").strip() in c)
            total_valid += ok
            print(f"[OK ] {url.split('?q=')[1]:24s} {dt:5.1f}s -> {ok}/{len(outs)} valid claims")
        except Exception as e:
            print(f"[FAIL] {url.split('?q=')[1]:24s} after {time.monotonic() - s:5.1f}s: "
                  f"{type(e).__name__}: {str(e)[:60]}")
    print(f"\nTotal valid claims from {len(SOURCES)} sources in {time.monotonic() - t0:.1f}s: "
          f"{total_valid} (pilot floor = 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
