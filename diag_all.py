"""Per-source extraction diagnostic across all pilot NASA sources.

For each URL: fetch via the hardened path, clean-excerpt it exactly as
extract_from_source_content does, run the production extractor ONCE (single draw),
and report whether it produced >=1 claim with a verbatim quote present in the excerpt.
This reveals which sources tiel-coder can ground vs. which abort the whole run.
"""

from pathlib import Path

from ai_video_factory.research_pipeline import (
    EXTRACTION_SCHEMA_VERSION,
    PROMPT_VERSION,
    SourceContent,
    _clean_excerpt,
    make_production_extractor,
    make_secure_http_transport,
)

TOPIC = "Water Beyond Earth: Where It Is and Why It Means We're Looking"
URLS = [
    "https://images-api.nasa.gov/search?q=Mars+ancient+riverbed",
    "https://images-api.nasa.gov/search?q=Europa+subsurface+ocean",
    "https://images-api.nasa.gov/search?q=Enceladus+plume",
    "https://images-api.nasa.gov/search?q=Lunar+south+pole+ice",
    "https://images-api.nasa.gov/search?q=Mars+water+ice+polar",
]


def one_source(transport, extractor, url) -> tuple[int, bool, str]:
    raw = transport(url)
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else (raw or "")
    chunk = _clean_excerpt(text[:20000], TOPIC)
    cleaned = chunk.text
    sc = SourceContent(url=url, title="", quality="high", content=cleaned, content_hash="")
    try:
        outs = extractor(TOPIC, [sc])
    except Exception as exc:  # noqa: BLE001
        return 0, False, f"{type(exc).__name__}: {str(exc)[:80]}"
    ok = 0
    for ext in outs:
        quote = getattr(ext, "quote", "") or ""
        if quote.strip() in cleaned:
            ok += 1
    return ok, ok > 0, f"{ok}/{len(outs)} claims quoted-in-excerpt"


def main() -> None:
    transport = make_secure_http_transport()
    extractor = make_production_extractor(endpoint_url="http://127.0.0.1:1234/v1/chat/completions")
    for url in URLS:
        ok, good, detail = one_source(transport, extractor, url)
        print(f"[{'OK ' if good else 'FAIL'}] {url}\n     {detail}", flush=True)


if __name__ == "__main__":
    main()
