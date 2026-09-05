# Iterative Improvement Prompt — AI Video Factory

Paste this into any future session to drive continuous improvement of the
video generation workflow toward production-grade YouTube documentaries.

---

**ITERATIVE IMPROVEMENT PROMPT FOR AI VIDEO FACTORY**

Context: This is an AI Video Factory repo at /home/summit/ai-video-factory. It
generates local YouTube documentaries using Remotion (compositions in
remotion/src/SyntheticVideo.tsx), FFmpeg, a Python pipeline
(src/ai_video_factory/video_pipeline.py), Piper TTS for narration, and stock
media from Wikimedia/NASA.

Goal: Iteratively improve the rendered video until it looks like production-grade
content that real creators upload to YouTube — not title cards with gradients,
but a polished documentary.

Quality bar (each iteration must demonstrably improve these):
- Transitions between scenes (cross-dissolve / zoom-through), no hard cuts
- Ken Burns pan/zoom on still images; slow, deliberate motion
- Intro sequence (branded title) + outro (credits); clean start/end
- Lower thirds and animated text reveals instead of static centered titles
- Color grading for a cinematic look
- Ambient music bed under narration + transition SFX
- Motion graphics where relevant (star fields, timelines, data points)

Methodology:
1. Make ONE focused improvement per iteration — don't rewrite everything at once.
2. Re-render a short segment (30–60s chunk) to test quickly before full render.
3. Compare before/after; keep the change only if it's clearly better and breaks nothing.
4. Run `uv run pytest -q` and `tsc --noEmit` after every change — both must stay green.
5. Re-render the full video with `run_video_pipeline` when a change is ready to ship.

Guardrails:
- Keep everything local (LM Studio worker, Piper TTS, stock media). No paid APIs, no publishing without explicit approval.
- Match existing code style; touch only what's needed.
- If a change breaks tests or the build, revert it and try another approach.
- Prefer incremental Remotion/CSS changes over rewriting SyntheticVideo.tsx from scratch.

Definition of done: The rendered video passes QC (10/10) AND visually matches
professional YouTube documentaries across all quality-bar items above. When
reached, commit the work and summarize what changed.
