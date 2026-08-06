import lightgbm as lgb
import numpy as np
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx


SEED = 20260805
ONNX_OPSET = 15


def _sample_data() -> tuple[np.ndarray, np.ndarray]:
    features = np.arange(320, dtype=np.float32).reshape(10, 32)
    targets = np.arange(10, dtype=np.float32)
    return features, targets


def test_scikit_learn_model_round_trips_through_onnxruntime() -> None:
    features, targets = _sample_data()
    model = RandomForestRegressor(n_estimators=2, random_state=SEED).fit(
        features, targets
    )

    onnx_model = to_onnx(model, features[:1], target_opset=ONNX_OPSET)
    session = ort.InferenceSession(onnx_model.SerializeToString())
    result = session.run(
        None,
        {session.get_inputs()[0].name: features[:1]},
    )

    assert result


def test_lightgbm_model_round_trips_through_onnxruntime() -> None:
    features, targets = _sample_data()
    model = lgb.LGBMRegressor(
        n_estimators=2,
        random_state=SEED,
        verbosity=-1,
    ).fit(features, targets)

    onnx_model = convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, 32]))],
        target_opset=ONNX_OPSET,
    )
    session = ort.InferenceSession(onnx_model.SerializeToString())
    result = session.run(
        None,
        {session.get_inputs()[0].name: features[:1]},
    )

    assert result
