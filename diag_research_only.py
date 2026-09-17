"""Replicate the daily_job research path exactly (no build/render) to diagnose the
~180s 'timed out' failure: measure total time, catch the precise exception, and report
per-source verified facts + drops. Mirrors _build_research_packages -> run_research."""

import sys
import time

from ai_video_factory.research_pipeline import (
    make_production_extractor,
    make_secure_http_transport,
    run_research,
)

TOPIC = "Water Beyond Earth: Where It Is and Why It Means We are Looking"
SOURCES = [
    "https://images-api.nasa.gov/search?q=Mars+ancient+riverbed",
    "https://images-api.nasa.gov/search?q=Mars+ancient+oceans",
    "https://images-api.nasa.gov/search?q=Mars+paleolakes+lakes",
]

def main() -> int:
    t0 = time.monotonic()
    extractor = make_production_extractor(endpoint_url="http://127.0.0.1:1234/v1/chat/completions")
    fetcher = make_secure_http_transport()
    try:
        res = run_research(
            TOPIC,
            SOURCES,
            production_extractor=extractor,
            content_fetcher=fetcher,
            droppable_urls=frozenset(SOURCES),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"research FAILED after {time.monotonic() - t0:.1f}s:", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    dt = time.monotonic() - t0
    print(f"research OK after {dt:.1f}s", file=sys.stderr)
    print(f"  verified_facts: {len(res.verified_facts)}", file=sys.stderr)
    print(f"  claims: {len(res.claims)}", file=sys.stderr)
    print(f"  dropped: {[d.url for d in res.dropped_sources]}", file=sys.stderr)
    for vf in res.verified_facts[:6]:
        print(f"    - {vf.claim_id}: {vf.text[:90]!r} -> {','.join(vf.source_ids)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
