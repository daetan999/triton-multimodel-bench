"""Generate paired ONNX Runtime and FIL models for Triton."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from xgboost import Booster, DMatrix, XGBRegressor


FEATURE_COUNT = 32
MAX_BATCH_SIZE = 1024
ONNX_OPSET = 15
TRAINING_ROWS = 256


class RepositoryValidationError(ValueError):
    """Raised when a generated model repository fails validation."""


def generate_repository(
    repository: Path | str,
    *,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Generate ``count`` equivalent ONNX/FIL model pairs and return a manifest."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    repository_path = Path(repository)
    repository_path.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for index in range(count):
        model_seed = seed + index
        model = _train_model(model_seed)
        onnx_name = f"onnx_model_{index:03d}"
        fil_name = f"fil_model_{index:03d}"
        onnx_path = repository_path / onnx_name / "1" / "model.onnx"
        fil_path = repository_path / fil_name / "1" / "xgboost.ubj"

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        fil_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_model = convert_xgboost(
            model,
            initial_types=[("input", FloatTensorType([None, FEATURE_COUNT]))],
            target_opset=ONNX_OPSET,
        )
        # onnxmltools otherwise assigns a random UUID as the graph name.
        onnx_model.graph.name = onnx_name
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, onnx_path)
        model.save_model(fil_path)

        onnx_input = onnx_model.graph.input[0].name
        onnx_output = onnx_model.graph.output[0].name
        (repository_path / onnx_name / "config.pbtxt").write_text(
            _onnx_config(onnx_name, onnx_input, onnx_output),
            encoding="utf-8",
        )
        (repository_path / fil_name / "config.pbtxt").write_text(
            _fil_config(fil_name),
            encoding="utf-8",
        )

        entries.append(
            {
                "index": index,
                "seed": model_seed,
                "onnx_model": onnx_name,
                "onnx_sha256": _sha256(onnx_path),
                "fil_model": fil_name,
                "fil_sha256": _sha256(fil_path),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "count": count,
        "seed": seed,
        "feature_count": FEATURE_COUNT,
        "onnx_opset": ONNX_OPSET,
        "max_batch_size": MAX_BATCH_SIZE,
        "models": entries,
    }
    (repository_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_repository(
    repository: Path | str,
    *,
    expected_count: int,
) -> dict[str, int | str]:
    """Validate every model pair and execute equivalent local predictions."""
    repository_path = Path(repository)
    manifest_path = repository_path / "manifest.json"
    if not manifest_path.is_file():
        raise RepositoryValidationError(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("count") != expected_count:
        raise RepositoryValidationError(
            f"expected {expected_count} model pairs, found {manifest.get('count')}"
        )
    entries = manifest.get("models")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise RepositoryValidationError("manifest model list does not match its count")

    onnx_directories = sorted(repository_path.glob("onnx_model_*"))
    fil_directories = sorted(repository_path.glob("fil_model_*"))
    if len(onnx_directories) != expected_count:
        raise RepositoryValidationError(
            f"expected {expected_count} ONNX model directories, "
            f"found {len(onnx_directories)}"
        )
    if len(fil_directories) != expected_count:
        raise RepositoryValidationError(
            f"expected {expected_count} FIL model directories, "
            f"found {len(fil_directories)}"
        )

    for entry in entries:
        _validate_pair(repository_path, entry)

    return {
        "model_pairs": expected_count,
        "onnx_models": expected_count,
        "fil_models": expected_count,
        "status": "valid",
    }


def _validate_pair(repository: Path, entry: dict[str, Any]) -> None:
    index = entry["index"]
    onnx_name = entry["onnx_model"]
    fil_name = entry["fil_model"]
    onnx_path = repository / onnx_name / "1" / "model.onnx"
    fil_path = repository / fil_name / "1" / "xgboost.ubj"
    onnx_config_path = repository / onnx_name / "config.pbtxt"
    fil_config_path = repository / fil_name / "config.pbtxt"

    for artifact in (onnx_path, fil_path, onnx_config_path, fil_config_path):
        if not artifact.is_file():
            raise RepositoryValidationError(f"missing artifact: {artifact}")
    if _sha256(onnx_path) != entry["onnx_sha256"]:
        raise RepositoryValidationError(f"ONNX hash mismatch for pair {index}")
    if _sha256(fil_path) != entry["fil_sha256"]:
        raise RepositoryValidationError(f"FIL hash mismatch for pair {index}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_input = onnx_model.graph.input[0].name
    onnx_output = onnx_model.graph.output[0].name
    if onnx_config_path.read_text(encoding="utf-8") != _onnx_config(
        onnx_name, onnx_input, onnx_output
    ):
        raise RepositoryValidationError(f"ONNX config mismatch for pair {index}")
    if fil_config_path.read_text(encoding="utf-8") != _fil_config(fil_name):
        raise RepositoryValidationError(f"FIL config mismatch for pair {index}")

    samples = np.random.default_rng(entry["seed"] ^ 0x5EED).normal(
        size=(16, FEATURE_COUNT)
    ).astype(np.float32)
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    onnx_predictions = session.run(
        [onnx_output],
        {onnx_input: samples},
    )[0].reshape(-1)
    booster = Booster()
    booster.load_model(fil_path)
    fil_predictions = booster.predict(DMatrix(samples)).reshape(-1)
    try:
        np.testing.assert_allclose(
            onnx_predictions,
            fil_predictions,
            rtol=1e-5,
            atol=1e-5,
        )
    except AssertionError as error:
        raise RepositoryValidationError(
            f"paired predictions differ for pair {index}: {error}"
        ) from error


def _train_model(seed: int) -> XGBRegressor:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(TRAINING_ROWS, FEATURE_COUNT)).astype(np.float32)
    weights = np.linspace(0.25, 2.0, num=8, dtype=np.float32)
    targets = (
        features[:, :8] @ weights
        + np.sin(features[:, 8] * np.float32(2.0))
        + rng.normal(scale=0.05, size=TRAINING_ROWS)
    ).astype(np.float32)
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=16,
        max_depth=4,
        learning_rate=0.15,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    ).fit(features, targets)


def _onnx_config(model_name: str, input_name: str, output_name: str) -> str:
    return f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {MAX_BATCH_SIZE}
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {FEATURE_COUNT} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]
'''


def _fil_config(model_name: str) -> str:
    return f'''name: "{model_name}"
backend: "fil"
max_batch_size: {MAX_BATCH_SIZE}
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {FEATURE_COUNT} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "xgboost_ubj" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "false" }}
  }}
]
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
