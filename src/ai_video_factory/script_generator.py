"""Script generation using local LM Studio model."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ai_video_factory.sanitization import sanitize_diagnostic


# System prompt for script writing
SCRIPT_SYSTEM_PROMPT = """You are an expert video scriptwriter and content strategist. Generate engaging, informative, and well-structured scripts for video production.

Your task is to create a sourced script based on trending topics, suitable for AI video factory production.

Requirements:
1. Use the trending topic as the central theme
2. Cite sources using markdown format [Source](URL)
3. Structure with clear sections: intro, main content, conclusion
4. Keep language conversational and engaging
5. Include timestamps for each scene
6. Provide visual suggestions for each segment
7. Write for a 60-90 second video (typically 150-250 words)

Output format:
- title: Creative video title
- narration: Full narration script
- scenes: List of scene objects with timing and visuals
- sources: List of source URLs used
- captions: SRT-style captions

Only output valid JSON, no markdown formatting."""


class ScriptGenerationError(RuntimeError):
    """Raised when a video script cannot be produced or validated.

    The pipeline treats this as a hard failure: the run is marked failed in
    the run manifest and no generic fallback video is produced silently.
    """


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output.

    Strips markdown code fences (```json ... ``` or ``` ... ```), then
    slices from the first ``{`` to the last ``}``. Raises
    ``ScriptGenerationError`` when no valid JSON object is found.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ScriptGenerationError("model output contained no JSON object")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise ScriptGenerationError(
            f"model output was not valid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise ScriptGenerationError("model output JSON was not an object")
    return parsed


def _validate_script_shape(result: dict[str, Any]) -> None:
    """Validate the fields the edit-document builder needs.

    Raises ``ScriptGenerationError`` on any violation so a malformed model
    response fails the run instead of producing a broken video.
    """
    title = result.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ScriptGenerationError("script is missing a non-empty 'title'")
    scenes = result.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ScriptGenerationError("script 'scenes' must be a non-empty list")
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ScriptGenerationError(f"scene {index} is not an object")
        if not scene.get("title") and not scene.get("caption"):
            raise ScriptGenerationError(
                f"scene {index} has neither 'title' nor 'caption'"
            )
        narration = scene.get("narration")
        if narration is not None and not isinstance(narration, str):
            raise ScriptGenerationError(f"scene {index} 'narration' is not a string")


def generate_script_with_lm_studio(
    topic_title: str,
    topic_description: str,
    source_url: str,
    api_url: str = "http://localhost:1234/v1/chat/completions",
    model: str = "qwen3.6-35b-a3b-udt-mtp",
    max_tokens: int = 8192,
    temperature: float = 0.7,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Generate a video script using LM Studio's local model.

    Raises ``ScriptGenerationError`` when the request fails or the response
    cannot be parsed/validated, unless ``allow_fallback`` is True, in which
    case a generic template script is returned. The default is fail-loud so
    unattended runs never silently produce placeholder videos.

    Reasoning models can spend their token budget thinking and return an
    empty message, so the request is retried once with a doubled budget
    when the model returns empty or unparseable content.
    """
    
    user_prompt = f"""Generate a video script about the following trending topic:

Title: {topic_title}
Description: {topic_description}
Source: {source_url}

Create a 60-90 second video script with:
1. An engaging title
2. A narration script with natural flow
3. Scene breakdown with timing (in frames at 30fps)
4. Visual suggestions for each scene
5. Proper citations
6. SRT-style captions

Make it engaging, factual, and visually descriptive."""

    def _build_payload(budget: int) -> bytes:
        return json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": budget,
            "temperature": temperature,
            "stream": False
        }).encode('utf-8')

    def _post(payload: bytes, budget: int) -> str:
        # Thinking models are slow: budget a ~6 tok/s floor so the call
        # cannot time out mid-stream on a CPU-only box.
        request_timeout = max(600, budget // 6)
        req = urllib.request.Request(
            api_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=request_timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        return result.get("choices", [{}])[0].get("message", {}).get("content") or ""

    def _use_fallback() -> dict[str, Any]:
        return _generate_fallback_script(topic_title, topic_description, source_url)

    last_error: ScriptGenerationError | None = None
    last_content = ""
    for budget in (max_tokens, max_tokens * 2):
        try:
            content = _post(_build_payload(budget), budget)
        except ScriptGenerationError:
            raise
        except Exception as error:
            if allow_fallback:
                # Explicitly requested fallback: produce a generic template.
                return _use_fallback()
            raise ScriptGenerationError(
                f"LM Studio script request failed: {error}"
            ) from error
        last_content = content
        if not content.strip():
            last_error = ScriptGenerationError(
                "LM Studio returned empty content "
                "(the model likely spent its token budget thinking)"
            )
            continue
        try:
            parsed = _extract_json(content)
            _validate_script_shape(parsed)
            return parsed
        except ScriptGenerationError as error:
            last_error = error
            continue

    preview = sanitize_diagnostic(last_content, max_chars=500)
    if allow_fallback:
        # Explicitly requested fallback: parse/validation failures also
        # degrade to the generic template rather than failing loud.
        return _use_fallback()
    detail = f": {last_error}" if last_error else ""
    raise ScriptGenerationError(
        f"LM Studio produced no usable script after 2 attempts{detail}; "
        f"model preview: {preview}"
    )


def _generate_fallback_script(
    topic_title: str,
    topic_description: str,
    source_url: str,
) -> dict[str, Any]:
    """Generate a basic script structure when LM Studio is unavailable."""
    
    # Create a basic script structure
    scenes = [
        {
            "id": "scene-0",
            "from_frame": 0,
            "duration_frames": 90,
            "title": topic_title,
            "caption": topic_description,
            "visual": "Animated title card with dynamic background",
            "narration": f"Today we're exploring: {topic_title}. {topic_description}",
        }
    ]
    
    return {
        "title": f"Trending: {topic_title}",
        "narration": f"{topic_description}\n\nSource: {source_url}",
        "scenes": scenes,
        "sources": [source_url],
        "captions": [
            {"start": 0, "end": 2.5, "text": f"{topic_title}"},
            {"start": 2.5, "end": 5.0, "text": topic_description},
        ],
        "generated_at": datetime.now().isoformat(),
        "method": "fallback"
    }


@dataclass
class ScriptOutput:
    title: str
    narration: str
    scenes: list[dict[str, Any]]
    sources: list[str]
    captions: list[dict[str, float]]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> str:
        return json.dumps({
            "title": self.title,
            "narration": self.narration,
            "scenes": self.scenes,
            "sources": self.sources,
            "captions": self.captions,
            "generated_at": self.generated_at,
        })


def generate_script(
    topic_title: str,
    topic_description: str,
    source_url: str,
    output_path: Path | None = None,
    use_local_model: bool = True,
    allow_fallback: bool = False,
) -> ScriptOutput:
    """Generate a video script about a trending topic.

    Args:
        topic_title: The trending topic title
        topic_description: Description of the topic
        source_url: Source URL for the topic
        output_path: Optional path to save the script JSON
        use_local_model: Whether to use LM Studio local model. Setting this
            to False explicitly requests the non-model path and still
            produces the generic template (this is an explicit choice, not
            a silent failure).
        allow_fallback: When True, a generic template script is returned if
            the model request fails. Defaults to False (fail loud).

    Returns:
        ScriptOutput object with the generated script

    Raises:
        ScriptGenerationError: when the model request fails or its response
            is invalid and ``allow_fallback`` is False.
    """

    if use_local_model:
        result = generate_script_with_lm_studio(
            topic_title, topic_description, source_url, allow_fallback=allow_fallback
        )
    else:
        result = _generate_fallback_script(topic_title, topic_description, source_url)
    
    # Parse the result
    script = ScriptOutput(
        title=result.get("title", f"Trending: {topic_title}"),
        narration=result.get("narration", ""),
        scenes=result.get("scenes", []),
        sources=result.get("sources", [source_url]),
        captions=result.get("captions", []),
        generated_at=result.get("generated_at", datetime.now().isoformat()),
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script.to_json())
    
    return script