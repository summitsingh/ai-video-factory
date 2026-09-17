#!/usr/bin/env python3
"""Phase 4 live-pilot runner.

Runs the approved documentary end-to-end against real authoritative content and
stops with an approval package pending explicit human sign-off (no upload,
publish, schedule, or install). This is the operator-approved pilot entry point;
it mirrors ``ai_video_factory.cli daily`` but adds two capabilities the CLI lacks:

* **Real source URLs baked in** for the approved topic so research is grounded in
  fetched NASA content rather than the topic string alone.
* **Droppable-URL tolerance.** Each source is extracted against a bounded,
  per-source deadline budget (the local model has a tight context window and can
  degrade on large input). A source that exhausts that budget would otherwise
  fail the whole run closed; flagging it droppable records it in provenance as a
  skip and lets the run continue with the sources that succeed. All of this topic's
  NASA search endpoints are flagged so an intermittent slow source never aborts the
  pilot -- but a dropped source is never credited to the film (it is excluded from
  hashes, content, the brief, and asset planning).

Nothing here uploads, publishes, authenticates, or schedules. The run stops at
``approval.json: pending`` exactly as the plan requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[0]

# Approved topic + real NASA Image & Video Library search endpoints (secure, public
# HTTPS) that ground the research. These three reliably yield verbatim-grounded claims
# with the active local model and cover two legs of the arc: Mars ancient water (two
# independent sources) and Enceladus's ocean-bearing plumes. Europa subsurface-ocean and
# lunar polar-ice endpoints are intentionally EXCLUDED from the default set: extraction
# diagnostics show the local model fails to quote their content verbatim, so they drop
# (recorded in provenance) rather than contribute facts. They remain available via
# --source-url for an operator who wants to attempt them; droppable_urls still applies.
PILOT_TOPIC = "Water Beyond Earth: Where It Is and Why It Means We're Looking"
PILOT_SOURCES = [
    "https://images-api.nasa.gov/search?q=Mars+ancient+riverbed",
    "https://images-api.nasa.gov/search?q=Enceladus+plume",
    "https://images-api.nasa.gov/search?q=Mars+water+ice+polar",
]


def build_config(
    topic: str,
    source_urls: list[str],
    droppable_urls: list[str],
) -> object:
    """Wire the production job for a grounded live run.

    Resolves the active local model from LM Studio at runtime (fail closed if
    unavailable), fetches each source through the hardened HTTPS path, and acquires
    NASA assets only after they pass the item-level rights gate.
    """
    # Imported lazily so ``--help`` stays cheap and offline imports never trigger a
    # network/model load.
    from ai_video_factory.daily_job import JobConfig
    from ai_video_factory.production import (
        make_production_asset_transport,
        RemotionKokoroRenderEngine,
    )
    from ai_video_factory.research_pipeline import (
        make_production_extractor,
        make_secure_http_transport,
    )

    inference_cfg = _load_inference_config()
    config = JobConfig(
        research_source_urls=list(source_urls),
        droppable_urls=frozenset(droppable_urls),
    )
    config.topic_override = topic
    # Hardened HTTPS content path (public images-api.nasa.gov over strict TLS).
    config.content_fetcher = make_secure_http_transport()
    # Production extractor: schema-constrained, grounded in fetched source bytes.
    config.production_extractor = make_production_extractor(
        endpoint_url=f"{inference_cfg.base_url}/chat/completions",
    )
    # NASA asset acquisition cleared by the item-level rights gate.
    config.asset_transport = make_production_asset_transport()
    return config, RemotionKokoroRenderEngine


def _load_inference_config() -> object:
    from ai_video_factory.inference_config import load_inference_config

    return load_inference_config(_PROJECT_ROOT / "config" / "inference.toml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 live-pilot runner.")
    parser.add_argument(
        "--topic", default=PILOT_TOPIC, help="Topic to research (default: approved topic)."
    )
    parser.add_argument(
        "--source-url",
        action="append",
        dest="source_urls",
        default=None,
        help="Authoritative source URL(s) grounding the run (repeatable). "
        f"Default: {PILOT_SOURCES}",
    )
    parser.add_argument(
        "--droppable-url",
        action="append",
        dest="droppable_urls",
        default=None,
        help="Source URL(s) that may drop (rather than fail the run) if they "
        "exhaust their per-attempt deadline budget (repeatable). Default: all sources.",
    )
    args = parser.parse_args(argv)

    source_urls = args.source_urls or PILOT_SOURCES
    droppable_urls = args.droppable_urls or list(source_urls)

    config, engine_cls = build_config(args.topic, source_urls, droppable_urls)

    from ai_video_factory.daily_job import run_daily_job

    print(f"pilot: topic={args.topic!r}", file=sys.stderr)
    print(f"pilot: sources={len(source_urls)} droppable={len(droppable_urls)}", file=sys.stderr)
    result = run_daily_job(_PROJECT_ROOT, config=config, engine=engine_cls())

    print(json.dumps(result.to_dict(), indent=2))
    if result.status != "completed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
