"""Tests for src/ai_video_factory/prompt_enhancer.py (L6)."""

from __future__ import annotations

from ai_video_factory import prompt_enhancer


def test_resolve_instructions_model_override():
    ltx2 = prompt_enhancer.resolve_instructions("ltx2")
    assert "LTX-2" in ltx2
    assert "150 words" in ltx2


def test_resolve_instructions_ltx25_override():
    text = prompt_enhancer.resolve_instructions("ltx2.5")
    assert "LTX-2.5" in text
    assert text != prompt_enhancer.BASE_T2V_CINEMATIC


def test_resolve_instructions_unknown_model_falls_back_to_base():
    assert (
        prompt_enhancer.resolve_instructions("something-else")
        == prompt_enhancer.BASE_T2V_CINEMATIC
    )


def test_base_template_has_cap_and_structure():
    base = prompt_enhancer.BASE_T2V_CINEMATIC
    assert "150 words" in base
    assert "single" in base and "paragraph" in base
    assert "camera" in base.lower()


def test_build_messages_shape():
    messages = prompt_enhancer.build_messages("a lighthouse", model="ltx2")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "a lighthouse"
    assert "LTX-2" in messages[0]["content"]


def _stub_ok(messages, max_tokens, **kwargs):
    _stub_ok.seen = {"max_tokens": max_tokens, "kwargs": kwargs,
                     "n_messages": len(messages)}
    return "A sweeping aerial shot over a misty harbor at dawn."


def test_enhance_prompt_calls_chat_fn():
    out = prompt_enhancer.enhance_prompt(
        "harbor", model="ltx2.5", chat_fn=_stub_ok, max_tokens=123,
        temperature=0.1,
    )
    assert out == "A sweeping aerial shot over a misty harbor at dawn."
    assert _stub_ok.seen["max_tokens"] == 123
    assert _stub_ok.seen["kwargs"]["temperature"] == 0.1
    assert _stub_ok.seen["n_messages"] == 2


def test_enhance_prompt_enforces_word_cap():
    long_text = "word " * 400
    out = prompt_enhancer.enhance_prompt(
        "x", chat_fn=lambda *a, **k: long_text.strip(),
    )
    assert len(out.split()) == prompt_enhancer.MAX_WORDS == 150


def test_enhance_prompt_falls_back_on_chat_error():
    def boom(*a, **k):
        raise RuntimeError("llm down")

    assert prompt_enhancer.enhance_prompt("raw idea", chat_fn=boom) == "raw idea"


def test_enhance_prompt_falls_back_on_empty_response():
    assert (
        prompt_enhancer.enhance_prompt("raw idea", chat_fn=lambda *a, **k: "  ")
        == "raw idea"
    )


def test_enhance_prompt_blank_input_returned_as_is():
    assert prompt_enhancer.enhance_prompt("   ", chat_fn=_stub_ok) == "   "


def test_enhance_prompt_passes_api_url():
    seen = {}

    def stub(messages, max_tokens, **kwargs):
        seen.update(kwargs)
        return "ok"

    prompt_enhancer.enhance_prompt(
        "x", chat_fn=stub, api_url="http://example:1234/v1/chat/completions"
    )
    assert seen["api_url"] == "http://example:1234/v1/chat/completions"
