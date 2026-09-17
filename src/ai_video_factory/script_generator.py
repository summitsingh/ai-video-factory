"""Script generation using local LM Studio model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import urllib.request


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


def generate_script_with_lm_studio(
    topic_title: str,
    topic_description: str,
    source_url: str,
    api_url: str = "http://localhost:1234/v1/chat/completions",
    model: str = "tiel-coder-35b-a3b-mtp",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate a video script using LM Studio's local model."""
    
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

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }).encode('utf-8')

    req = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            return json.loads(content)
    except Exception as e:
        # Fallback to local generation if API fails
        return _generate_fallback_script(topic_title, topic_description, source_url)


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
) -> ScriptOutput:
    """Generate a video script about a trending topic.
    
    Args:
        topic_title: The trending topic title
        topic_description: Description of the topic
        source_url: Source URL for the topic
        output_path: Optional path to save the script JSON
        use_local_model: Whether to use LM Studio local model
        
    Returns:
        ScriptOutput object with the generated script
    """
    
    if use_local_model:
        try:
            result = generate_script_with_lm_studio(
                topic_title, topic_description, source_url
            )
        except Exception:
            result = _generate_fallback_script(topic_title, topic_description, source_url)
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