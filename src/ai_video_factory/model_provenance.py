"""Contained, read-only identity for local LM Studio model files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.inference_models import ModelIdentity
from ai_video_factory.lm_studio import LmStudioModel, LmStudioSnapshot
from ai_video_factory.sanitization import sanitize_diagnostic


_CACHE_SCHEMA_VERSION = 1
_HASH_BLOCK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
Hasher = Callable[[Path], str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as model_file:
        while block := model_file.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


class ModelDigestCache:
    """Cache model digests only after independently re-checking containment."""

    def __init__(
        self,
        cache_directory: Path,
        *,
        model_root: Path,
        identifier: str | None = None,
        hasher: Hasher = _sha256_file,
    ) -> None:
        self.cache_directory = Path(cache_directory)
        self.model_root = Path(model_root)
        self.identifier = identifier
        self.hasher = hasher

    def identity(self, model: LmStudioModel) -> ModelIdentity:
        """Return a SHA-256 identity without modifying or relocating the model."""
        path, relative_path, file_stat = self._contained_file(model)
        if file_stat.st_size != model.size_bytes:
            companion_size = self._direct_companion_size(path)
            if file_stat.st_size + companion_size != model.size_bytes:
                raise ValueError(
                    "LM Studio inventory size does not match the contained model package"
                )

        cache_path = self._cache_path(relative_path)
        cached_digest = self._cached_digest(
            cache_path,
            relative_path=relative_path,
            size_bytes=file_stat.st_size,
            mtime_ns=file_stat.st_mtime_ns,
        )
        digest = cached_digest or self.hasher(path)
        if cached_digest is None:
            after_hash = path.stat()
            if (
                not stat.S_ISREG(after_hash.st_mode)
                or after_hash.st_size != file_stat.st_size
                or after_hash.st_mtime_ns != file_stat.st_mtime_ns
            ):
                raise ValueError("LM Studio model changed while its digest was being computed")
            self._write_cache(
                cache_path,
                relative_path=relative_path,
                size_bytes=file_stat.st_size,
                mtime_ns=file_stat.st_mtime_ns,
                sha256=digest,
            )

        return model.identity(self.identifier or model.model_key).model_copy(
            update={"relative_path": relative_path, "size_bytes": file_stat.st_size, "sha256": digest}
        )

    def _direct_companion_size(self, primary: Path) -> int:
        root = self.model_root.resolve(strict=False)
        try:
            candidates = tuple(primary.parent.glob("mmproj*.gguf"))
        except OSError as exc:
            raise ValueError("LM Studio model companions could not be inspected") from exc

        total = 0
        for candidate in candidates:
            if candidate == primary:
                continue
            try:
                companion_stat = candidate.lstat()
                if stat.S_ISLNK(companion_stat.st_mode):
                    raise ValueError(
                        "LM Studio model companion must be a regular non-symlink file"
                    )
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except ValueError:
                raise
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    "LM Studio model companion must be contained by models_directory"
                ) from exc
            if resolved.parent != primary.parent:
                raise ValueError(
                    "LM Studio model companion must be directly beside the primary model"
                )
            if not stat.S_ISREG(companion_stat.st_mode):
                raise ValueError(
                    "LM Studio model companion must be a regular non-symlink file"
                )
            total += companion_stat.st_size
        return total

    def _contained_file(self, model: LmStudioModel) -> tuple[Path, str, os.stat_result]:
        root = self.model_root.resolve(strict=False)
        relative = Path(model.relative_path)
        if relative.is_absolute():
            raise ValueError("LM Studio model path must be contained by models_directory")
        candidate = root / relative
        if candidate.is_symlink() or Path(model.path).is_symlink():
            raise ValueError("LM Studio model must be a regular non-symlink file")
        try:
            resolved_candidate = candidate.resolve(strict=True)
            resolved_model = Path(model.path).resolve(strict=True)
            resolved_candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("LM Studio model path must be contained by models_directory") from exc
        if resolved_candidate != resolved_model:
            raise ValueError("LM Studio model path must be contained by models_directory")
        try:
            file_stat = resolved_candidate.stat()
        except OSError as exc:
            raise ValueError("LM Studio model must be a regular non-symlink file") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("LM Studio model must be a regular non-symlink file")
        return resolved_candidate, relative.as_posix(), file_stat

    def _cache_path(self, relative_path: str) -> Path:
        key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        return self.cache_directory / f"{key}.json"

    @staticmethod
    def _cached_digest(
        path: Path,
        *,
        relative_path: str,
        size_bytes: int,
        mtime_ns: int,
    ) -> str | None:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict) or set(record) != {
            "schema_version", "relative_path", "size_bytes", "mtime_ns", "sha256",
        }:
            return None
        digest = record.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            return None
        if not isinstance(record["relative_path"], str) or any(
            isinstance(record[field], bool) or not isinstance(record[field], int)
            for field in ("schema_version", "size_bytes", "mtime_ns")
        ):
            return None
        if (
            record["schema_version"] != _CACHE_SCHEMA_VERSION
            or record["relative_path"] != relative_path
            or record["size_bytes"] != size_bytes
            or record["mtime_ns"] != mtime_ns
        ):
            return None
        return digest

    def _write_cache(
        self,
        path: Path,
        *,
        relative_path: str,
        size_bytes: int,
        mtime_ns: int,
        sha256: str,
    ) -> None:
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError("model SHA-256 digest is invalid")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
        record = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "sha256": sha256,
        }
        try:
            with temporary.open("x", encoding="utf-8") as cache_file:
                cache_file.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
                cache_file.flush()
                os.fsync(cache_file.fileno())
            temporary.replace(path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def capability_inputs(
    config: InferenceConfig,
    snapshot: LmStudioSnapshot,
    identity: ModelIdentity,
    corpus_version: str,
) -> dict[str, object]:
    """Return the sanitized, deterministic inputs for a capability run."""
    inventory_size = snapshot.configured_model.size_bytes
    companion_size = inventory_size - identity.size_bytes
    if companion_size < 0:
        raise ValueError("LM Studio inventory size is smaller than the primary model file")
    model = identity.model_dump(mode="json")
    model.update(
        {
            "inventory_size_bytes": inventory_size,
            "companion_size_bytes": companion_size,
        }
    )
    return {
        "config": config.model_dump(mode="json"),
        "cli": sanitize_diagnostic(snapshot.cli_help),
        "runtime": sanitize_diagnostic(snapshot.runtimes),
        "amd_survey": sanitize_diagnostic(snapshot.runtime_survey),
        "model": model,
        "corpus_version": corpus_version,
    }
