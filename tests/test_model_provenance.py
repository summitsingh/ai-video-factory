from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai_video_factory.inference_config import InferenceConfig
from ai_video_factory.lm_studio import LmStudioModel, LmStudioSnapshot
from ai_video_factory.inference_models import InferenceCheck
from ai_video_factory.model_provenance import ModelDigestCache, capability_inputs
from ai_video_factory.run_store import fingerprint_inputs


class RecordingHasher:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()


def model_record(
    path: Path,
    *,
    root: Path | None = None,
    size_bytes: int | None = None,
    relative_path: str | None = None,
) -> LmStudioModel:
    return LmStudioModel(
        model_key="local-model",
        path=path,
        relative_path=relative_path or path.relative_to(root or path.parent).as_posix(),
        size_bytes=path.stat().st_size if size_bytes is None else size_bytes,
    )


def cache_fixture(tmp_path: Path) -> tuple[ModelDigestCache, Path, RecordingHasher]:
    model_root = tmp_path / "models"
    model = model_root / "publisher" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-bytes")
    hasher = RecordingHasher()
    return (
        ModelDigestCache(tmp_path / "cache", model_root=model_root, hasher=hasher),
        model,
        hasher,
    )


def test_hashes_model_and_atomically_caches_identity(tmp_path: Path) -> None:
    model = tmp_path / "models" / "publisher" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-bytes")
    cache = ModelDigestCache(tmp_path / "cache", model_root=tmp_path / "models")

    identity = cache.identity(model_record(model, root=tmp_path / "models"))

    assert identity.sha256 == hashlib.sha256(b"model-bytes").hexdigest()
    assert not list((tmp_path / "cache").glob("*.tmp"))
    assert json.loads(next((tmp_path / "cache").glob("*.json")).read_text()) == {
        "schema_version": 1,
        "relative_path": "publisher/model.gguf",
        "size_bytes": len(b"model-bytes"),
        "mtime_ns": model.stat().st_mtime_ns,
        "sha256": hashlib.sha256(b"model-bytes").hexdigest(),
    }


def test_primary_only_inventory_size_preserves_primary_identity(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)

    identity = cache.identity(model_record(model, root=tmp_path / "models"))

    assert identity.size_bytes == len(b"model-bytes")
    assert hasher.calls == [model]


def test_accepts_exact_primary_plus_direct_mmproj_package(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    companion = model.parent / "mmproj-F16.gguf"
    companion.write_bytes(b"vision-projector")
    inventory_size = model.stat().st_size + companion.stat().st_size

    identity = cache.identity(
        model_record(
            model,
            root=tmp_path / "models",
            size_bytes=inventory_size,
        )
    )

    assert identity.size_bytes == model.stat().st_size
    assert identity.sha256 == hashlib.sha256(b"model-bytes").hexdigest()
    assert hasher.calls == [model]


def test_reuses_digest_only_when_path_size_and_mtime_match(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    record = model_record(model, root=tmp_path / "models")

    cache.identity(record)
    cache.identity(record)
    assert hasher.calls == [model]

    model.write_bytes(b"changed")
    cache.identity(model_record(model, root=tmp_path / "models"))

    assert hasher.calls == [model, model]


def test_rejects_symlink_or_path_escape(tmp_path: Path) -> None:
    cache, model, _hasher = cache_fixture(tmp_path)
    escaped = LmStudioModel(
        model_key="local-model",
        path=Path("../../secret"),
        relative_path="../../secret",
        size_bytes=0,
    )

    with pytest.raises(ValueError, match="contained"):
        cache.identity(escaped)

    alias = model.parent / "alias.gguf"
    alias.symlink_to(model)
    with pytest.raises(ValueError, match="non-symlink"):
        cache.identity(model_record(alias, root=tmp_path / "models"))


def test_rejects_inventory_size_that_disagrees_with_file(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)

    with pytest.raises(ValueError, match="size"):
        cache.identity(model_record(model, root=tmp_path / "models", size_bytes=1))

    assert not hasher.calls


def test_rejects_unexplained_package_size_mismatch(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    companion = model.parent / "mmproj-F16.gguf"
    companion.write_bytes(b"vision-projector")

    with pytest.raises(ValueError, match="size"):
        cache.identity(
            model_record(
                model,
                root=tmp_path / "models",
                size_bytes=model.stat().st_size + companion.stat().st_size + 1,
            )
        )

    assert not hasher.calls


def test_rejects_symlinked_mmproj_companion(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    projector = model.parent / "projector.gguf"
    projector.write_bytes(b"vision-projector")
    companion = model.parent / "mmproj-F16.gguf"
    companion.symlink_to(projector)

    with pytest.raises(ValueError, match="non-symlink"):
        cache.identity(
            model_record(
                model,
                root=tmp_path / "models",
                size_bytes=model.stat().st_size + projector.stat().st_size,
            )
        )

    assert not hasher.calls


def test_does_not_count_unrelated_sibling_as_package_companion(tmp_path: Path) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    unrelated = model.parent / "adapter.gguf"
    unrelated.write_bytes(b"unrelated")

    with pytest.raises(ValueError, match="size"):
        cache.identity(
            model_record(
                model,
                root=tmp_path / "models",
                size_bytes=model.stat().st_size + unrelated.stat().st_size,
            )
        )

    assert not hasher.calls


@pytest.mark.parametrize(
    "cache_record",
    [
        "not-json",
        json.dumps(
            {
                "schema_version": 1,
                "relative_path": "publisher/model.gguf",
                "size_bytes": len(b"model-bytes"),
                "mtime_ns": 0,
                "sha256": "0" * 63,
            }
        ),
    ],
)
def test_rehashes_when_cache_record_is_malformed_or_invalid(
    tmp_path: Path, cache_record: str,
) -> None:
    cache, model, hasher = cache_fixture(tmp_path)
    record = model_record(model, root=tmp_path / "models")
    cache.identity(record)
    cache_file = next((tmp_path / "cache").glob("*.json"))
    cache_file.write_text(cache_record)
    hasher.calls.clear()

    identity = cache.identity(record)

    assert identity.sha256 == hashlib.sha256(b"model-bytes").hexdigest()
    assert hasher.calls == [model]


def test_cache_replacement_uses_a_same_directory_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, model, _hasher = cache_fixture(tmp_path)
    replaced: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def record_replace(source: Path, destination: Path) -> Path:
        replaced.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", record_replace)

    cache.identity(model_record(model, root=tmp_path / "models"))

    assert len(replaced) == 1
    temporary, destination = replaced[0]
    assert temporary.parent == destination.parent == tmp_path / "cache"
    assert temporary.name.endswith(".tmp")
    assert destination.suffix == ".json"


def config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "schema_version": 1,
            "backend": "lm_studio",
            "base_url": "http://127.0.0.1:1234/v1",
            "lms_binary": "lms",
            "model_key": "local-model",
            "identifier": "local-model-identity",
            "context_length": 65_536,
            "gpu": "max",
            "parallel": 1,
            "ttl_seconds": 3_600,
            "minimum_available_memory_gib": 40,
            "models_directory": str(tmp_path / "models"),
        }
    )


def snapshot(model: LmStudioModel) -> LmStudioSnapshot:
    return LmStudioSnapshot(
        cli_help="lms 0.0.47",
        runtimes="rocm-runtime 2.31.2 token=hidden",
        runtime_survey="AMD Radeon 8060S\nsecret=hidden",
        server_status="running",
        server_running=True,
        models=(model,),
        loaded_identifiers=("local-model-identity",),
        configured_model=model,
        configured_model_loaded=True,
        checks={"cli": InferenceCheck(status="ready", detail="lms 0.0.47")},
    )


def test_capability_inputs_have_a_deterministic_sanitized_fingerprint(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_path = model_root / "publisher" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model-bytes")
    model = model_record(model_path, root=model_root)
    identity = ModelDigestCache(tmp_path / "cache", model_root=model_root).identity(model)
    inputs = capability_inputs(config(tmp_path), snapshot(model), identity, "lm-studio-capability-v4")

    assert inputs["corpus_version"] == "lm-studio-capability-v4"
    assert inputs["runtime"] == "rocm-runtime 2.31.2 token=[REDACTED]"
    assert inputs["amd_survey"] == "AMD Radeon 8060S\nsecret=[REDACTED]"
    assert inputs["model"]["sha256"] == hashlib.sha256(b"model-bytes").hexdigest()
    assert inputs["model"]["size_bytes"] == len(b"model-bytes")
    assert inputs["model"]["inventory_size_bytes"] == len(b"model-bytes")
    assert inputs["model"]["companion_size_bytes"] == 0
    assert fingerprint_inputs(inputs) == fingerprint_inputs(
        capability_inputs(config(tmp_path), snapshot(model), identity, "lm-studio-capability-v4")
    )


def test_capability_fingerprint_changes_when_companion_size_changes(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    model_path = model_root / "publisher" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model-bytes")
    companion = model_path.parent / "mmproj-F16.gguf"
    companion.write_bytes(b"vision")
    cache = ModelDigestCache(tmp_path / "cache", model_root=model_root)

    first_model = model_record(
        model_path,
        root=model_root,
        size_bytes=model_path.stat().st_size + companion.stat().st_size,
    )
    first_identity = cache.identity(first_model)
    first = capability_inputs(
        config(tmp_path),
        snapshot(first_model),
        first_identity,
        "lm-studio-capability-v4",
    )

    companion.write_bytes(b"vision-expanded")
    second_model = model_record(
        model_path,
        root=model_root,
        size_bytes=model_path.stat().st_size + companion.stat().st_size,
    )
    second_identity = cache.identity(second_model)
    second = capability_inputs(
        config(tmp_path),
        snapshot(second_model),
        second_identity,
        "lm-studio-capability-v4",
    )

    assert first["model"]["inventory_size_bytes"] == len(b"model-bytesvision")
    assert first["model"]["companion_size_bytes"] == len(b"vision")
    assert second["model"]["inventory_size_bytes"] == len(
        b"model-bytesvision-expanded"
    )
    assert second["model"]["companion_size_bytes"] == len(b"vision-expanded")
    assert first_identity.sha256 == second_identity.sha256
    assert fingerprint_inputs(first) != fingerprint_inputs(second)
