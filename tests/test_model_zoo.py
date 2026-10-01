import json
from pathlib import Path

import pytest

from triton_benchmark.model_zoo import (
    RepositoryValidationError,
    generate_repository,
    validate_repository,
)


def test_generate_repository_creates_one_complete_model_pair(tmp_path: Path) -> None:
    repository = tmp_path / "models"

    manifest = generate_repository(repository, count=1, seed=20260805)

    assert manifest["count"] == 1
    assert manifest["seed"] == 20260805
    assert (repository / "onnx_model_000" / "1" / "model.onnx").is_file()
    assert (repository / "onnx_model_000" / "config.pbtxt").is_file()
    assert (repository / "fil_model_000" / "1" / "xgboost.ubj").is_file()
    assert (repository / "fil_model_000" / "config.pbtxt").is_file()

    fil_config = (repository / "fil_model_000" / "config.pbtxt").read_text()
    assert 'key: "output_class"' in fil_config
    assert 'value: { string_value: "false" }' in fil_config
    assert 'key: "is_classifier"' not in fil_config

    stored_manifest = json.loads((repository / "manifest.json").read_text())
    assert stored_manifest == manifest


def test_validate_repository_executes_each_model_pair(tmp_path: Path) -> None:
    repository = tmp_path / "models"
    generate_repository(repository, count=2, seed=20260805)

    report = validate_repository(repository, expected_count=2)

    assert report == {
        "model_pairs": 2,
        "onnx_models": 2,
        "fil_models": 2,
        "status": "valid",
    }


def test_generate_repository_is_deterministic(tmp_path: Path) -> None:
    first = generate_repository(tmp_path / "first", count=2, seed=20260805)
    second = generate_repository(tmp_path / "second", count=2, seed=20260805)

    assert first == second


def test_validate_repository_rejects_modified_artifact(tmp_path: Path) -> None:
    repository = tmp_path / "models"
    generate_repository(repository, count=1, seed=20260805)
    artifact = repository / "onnx_model_000" / "1" / "model.onnx"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(RepositoryValidationError, match="ONNX hash mismatch"):
        validate_repository(repository, expected_count=1)
