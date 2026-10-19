"""CPU-only preflight validation for AI Video Factory (Q4).

Concept borrowed from WanGP's ``wangp_validate`` tool: validate everything
that can fail BEFORE any expensive stage runs, without loading models,
running inference, or touching the network. A 25-minute render should never
die at minute 20 on a missing file, a missing API key, or a misaligned
canvas.

This module is a clean-room reimplementation of the *concept* for this
pipeline's own stages (stock credentials, hero-clip backend, resolution
grid, music bed, caption fonts, inference config). No WanGP code is used
here. (WanGP is under the WanGP Community License 2.0; see
``/home/summit/wangp/LICENSE.txt``.)

Severity model: ``fail`` means the run cannot succeed as configured;
``warn`` means it will run but degraded (a fallback exists); ``pass``
means the check is satisfied. ``PreflightReport.ok`` is True when there
are no failures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

# LTX-family models require spatial dims on a 32px grid (hero_video rounds up
# and center-crops; preflight reports the adjustment instead of failing).
HERO_GRID_STRIDE = 32

# System font files the pipeline's own caption/thumbnail renderers accept
# before falling back to Pillow's bundled bitmap font.
_CAPTION_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

_STOCK_KEY_PROVIDERS = (
    ("PEXELS_API_KEY", "pexels"),
    ("PIXABAY_API_KEY", "pixabay"),
)


@dataclass
class CheckResult:
    """One named preflight check."""

    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


@dataclass
class PreflightReport:
    """The full preflight outcome."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no check failed (warnings are allowed)."""
        return all(c.status != STATUS_FAIL for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_WARN]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def check_stock_api_keys(env: Mapping[str, str] | None = None) -> CheckResult:
    """API keys for the stock-footage providers (presence only, no network)."""
    mapping = _env(env)
    configured = [
        provider
        for var, provider in _STOCK_KEY_PROVIDERS
        if str(mapping.get(var, "")).strip()
    ]
    if not configured:
        return CheckResult(
            name="stock_api_keys",
            status=STATUS_FAIL,
            detail=(
                "neither PEXELS_API_KEY nor PIXABAY_API_KEY is set; the "
                "stock-footage stage will find nothing. Presence only is "
                "checked here; key validity is not tested (no network in "
                "preflight)."
            ),
        )
    return CheckResult(
        name="stock_api_keys",
        status=STATUS_PASS,
        detail=(
            "stock providers configured: " + ", ".join(configured)
            + " (presence only; validity not tested)"
        ),
    )


def check_hero_backend(
    repo_root: Path, env: Mapping[str, str] | None = None
) -> CheckResult:
    """LTX hero-clip backend: venv python, worker script, model weights."""
    mapping = _env(env)
    root = Path(repo_root)
    venv_python = Path(
        mapping.get("LTX2_VENV", str(root / ".venv-ltx2"))
    ) / "bin" / "python"
    worker = root / "scripts" / "generate_ltx2.py"
    model_dir = Path(mapping.get("LTX2_MODEL_DIR", str(root / "models" / "LTX-2")))
    transformer_dir = model_dir / "transformer"

    problems: list[str] = []
    if not venv_python.is_file():
        problems.append(f"backend venv python missing: {venv_python}")
    if not worker.is_file():
        problems.append(f"worker script missing: {worker}")
    if not transformer_dir.is_dir() or not any(
        transformer_dir.glob("*.safetensors")
    ):
        problems.append(
            f"LTX weights not found under {transformer_dir}/ "
            "(run: python scripts/download_ltx2.py)"
        )
    if problems:
        return CheckResult(
            name="hero_backend",
            status=STATUS_WARN,
            detail=(
                "hero-clip backend unavailable (" + "; ".join(problems) + "). "
                "Hero clips will be skipped; stock-first scenes are unaffected."
            ),
        )
    return CheckResult(
        name="hero_backend",
        status=STATUS_PASS,
        detail=f"LTX backend ready (venv={venv_python.parent.parent}, model={model_dir})",
    )


def check_resolution_grid(
    width: int, height: int, stride: int = HERO_GRID_STRIDE
) -> CheckResult:
    """Requested dims vs the model grid; reports the adjusted canvas."""
    if width <= 0 or height <= 0:
        return CheckResult(
            name="resolution_grid",
            status=STATUS_FAIL,
            detail=f"invalid dimensions: {width}x{height}",
        )
    snapped_w = (width + stride - 1) // stride * stride
    snapped_h = (height + stride - 1) // stride * stride
    if (snapped_w, snapped_h) == (width, height):
        return CheckResult(
            name="resolution_grid",
            status=STATUS_PASS,
            detail=f"{width}x{height} already on the {stride}px grid",
        )
    return CheckResult(
        name="resolution_grid",
        status=STATUS_WARN,
        detail=(
            f"requested {width}x{height} is off the {stride}px grid; "
            f"generation rounds up to {snapped_w}x{snapped_h} and "
            "center-crops back (hero_video behavior)"
        ),
    )


def check_music_bed(
    repo_root: Path,
    data_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CheckResult:
    """A real music bed resolves (curated library, env override, or cache)."""
    mapping = _env(env)
    root = Path(repo_root)
    sources: list[str] = []

    override = str(mapping.get("MUSIC_BED_PATH", "")).strip()
    if override:
        if Path(override).is_file():
            sources.append(f"MUSIC_BED_PATH={override}")
        else:
            return CheckResult(
                name="music_bed",
                status=STATUS_FAIL,
                detail=f"MUSIC_BED_PATH points at a missing file: {override}",
            )

    library = sorted((root / "assets" / "music").glob("*-bed.*"))
    if library:
        sources.append(f"curated library: {library[0].name} (+{len(library) - 1} more)")

    if data_root is not None:
        cached = sorted((Path(data_root) / "cache" / "music").glob("*"))
        cached = [p for p in cached if p.is_file()]
        if cached:
            sources.append(f"cache: {cached[0].name} (+{len(cached) - 1} more)")

    if sources:
        return CheckResult(
            name="music_bed",
            status=STATUS_PASS,
            detail="music bed resolves via " + "; ".join(sources),
        )
    return CheckResult(
        name="music_bed",
        status=STATUS_WARN,
        detail=(
            "no curated music bed found (assets/music/*-bed.*, MUSIC_BED_PATH, "
            "or data/cache/music/); the mix falls back to the procedural "
            "ambient drone"
        ),
    )


def check_caption_fonts() -> CheckResult:
    """A real TTF is available for caption/thumbnail text rendering."""
    found = [p for p in _CAPTION_FONT_CANDIDATES if Path(p).is_file()]
    if found:
        return CheckResult(
            name="caption_fonts",
            status=STATUS_PASS,
            detail=f"font available: {found[0]}",
        )
    return CheckResult(
        name="caption_fonts",
        status=STATUS_WARN,
        detail=(
            "no system TTF found in the known caption font paths; text "
            "rendering falls back to Pillow's bundled bitmap font"
        ),
    )


def check_inference_config(repo_root: Path) -> CheckResult:
    """LM Studio inference config exists and parses (script generation)."""
    config_path = Path(repo_root) / "config" / "inference.toml"
    if not config_path.is_file():
        return CheckResult(
            name="inference_config",
            status=STATUS_WARN,
            detail=f"missing {config_path}; script generation needs LM Studio config",
        )
    try:
        import tomllib

        with config_path.open("rb") as fh:
            tomllib.load(fh)
    except Exception as error:  # noqa: BLE001 - any parse failure is a warn
        return CheckResult(
            name="inference_config",
            status=STATUS_WARN,
            detail=f"{config_path} does not parse as TOML: {error}",
        )
    return CheckResult(
        name="inference_config",
        status=STATUS_PASS,
        detail=f"inference config parses: {config_path}",
    )


def run_preflight(
    repo_root: Path,
    data_root: Path | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    stride: int = HERO_GRID_STRIDE,
    env: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Run every preflight check. CPU-only; never loads models or networks."""
    root = Path(repo_root)
    data = Path(data_root) if data_root is not None else root / "data"
    return PreflightReport(
        checks=[
            check_stock_api_keys(env),
            check_hero_backend(root, env),
            check_resolution_grid(width, height, stride),
            check_music_bed(root, data, env),
            check_caption_fonts(),
            check_inference_config(root),
        ]
    )


def format_report(report: PreflightReport) -> str:
    """Human-readable preflight report for CLI output."""
    lines = ["Preflight report:"]
    for check in report.checks:
        marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[check.status]
        lines.append(f"  [{marker}] {check.name}: {check.detail}")
    verdict = (
        "preflight OK" if report.ok else "preflight FAILED: fix the FAIL items above"
    )
    lines.append(verdict)
    return "\n".join(lines)
