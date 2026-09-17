"""Diagnose why the production extractor returns no verbatim-evidence claims.

Fetches one real NASA source through the hardened path, cleans its excerpt exactly
as extract_from_source_content does, runs the production extractor against it, and
prints each returned claim + quote plus whether the quote is a verbatim substring of
the cleaned excerpt (the central grounding gate's check).
"""

from pathlib import Path

from ai_video_factory.research_pipeline import (
    EXTRACTION_SCHEMA_VERSION,
    PROMPT_VERSION,
    SourceContent,
    _clean_excerpt,
    make_production_extractor,
    make_secure_http_transport,
    resolve_active_model,
)

TOPIC = "Water Beyond Earth: Where It Is and Why It Means We're Looking"
URL = "https://images-api.nasa.gov/search?q=Mars+ancient+riverbed"


def main() -> None:
    transport = make_secure_http_transport()
    model = resolve_active_model(endpoint_url="http://127.0.0.1:1234/v1/chat/completions")
    print(f"resolved model: {model}", flush=True)

    raw = transport(URL)
    content = getattr(raw, "decode", lambda: raw)() if isinstance(raw, (bytes, bytearray)) else raw
    text = content if isinstance(content, str) else content.decode("utf-8", "replace")
    print(f"raw fetched bytes: {len(text)}", flush=True)

    chunk = _clean_excerpt(text[:20000], TOPIC)  # cap to keep excerpt bounded like prod
    cleaned = chunk.text
    print(f"cleaned excerpt chars: {len(cleaned)}", flush=True)
    print("--- first 600 chars of cleaned excerpt ---")
    print(cleaned[:600], flush=True)
    print("--- end excerpt ---", flush=True)

    extractor = make_production_extractor(
        endpoint_url="http://127.0.0.1:1234/v1/chat/completions",
        extractor_name="lm-studio-production",
        extractor_version="1.0",
        prompt_version=PROMPT_VERSION,
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
    )
    sc = SourceContent(url=URL, title="Mars ancient riverbed", quality="high", content=cleaned, content_hash="")
    print(f"excerpt contains 'meandering river'? {'meandering river' in cleaned}", flush=True)
    try:
        outs = extractor(TOPIC, [sc])
    except Exception as exc:  # noqa: BLE001
        print(f"extractor raised: {type(exc).__name__}: {exc}", flush=True)
        return

    print(f"extractor returned {len(outs)} record(s):", flush=True)
    for i, ext in enumerate(outs):
        quote = getattr(ext, "quote", None)
        claim = getattr(ext, "claim", None)
        cls = getattr(ext, "classification", "?")
        matches = isinstance(quote, str) and quote.strip() in cleaned
        print(f"[{i}] classification={cls!r}", flush=True)
        print(f"    claim={claim!r}", flush=True)
        print(f"    quote={quote!r}", flush=True)
        print(f"    quote-in-excerpt? {matches} (len quote={len(quote or '')})", flush=True)


if __name__ == "__main__":
    main()
