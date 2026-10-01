import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from triton_benchmark.matrix_runner import MatrixRunSpec, execute_matrix
from triton_benchmark.model_zoo import generate_repository
from triton_benchmark.triton_runtime import (
    GpuTelemetry,
    ProcessTelemetry,
    TritonGrpcAdapter,
    TritonServer,
    model_names,
)


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 123
        self.returncode = None
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: int) -> int:
        self.waited = True
        self.returncode = 0
        return 0


class FakeResponse:
    status = 200

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return json.dumps(self.payload).encode()


def test_triton_server_starts_checks_models_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()
    monkeypatch.setattr(
        "triton_benchmark.triton_runtime.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "triton_benchmark.triton_runtime.urllib.request.urlopen",
        lambda *args, **kwargs: FakeResponse({}),
    )
    monkeypatch.setattr(
        TritonServer,
        "_ready_model_names",
        staticmethod(lambda: {"onnx_model_000"}),
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "triton_benchmark.triton_runtime.os.killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )
    server = TritonServer(
        command=("tritonserver",),
        expected_models=("onnx_model_000",),
        log_path=tmp_path / "server.log",
        readiness_timeout_seconds=1,
    )

    with server:
        assert (tmp_path / "server.log").exists()

    assert process.waited is True
    assert killed


def test_repository_index_only_returns_ready_version_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {"name": "ready", "version": "1", "state": "READY"},
        {"name": "wrong-version", "version": "2", "state": "READY"},
        {"name": "loading", "version": "1", "state": "LOADING"},
    ]
    monkeypatch.setattr(
        "triton_benchmark.triton_runtime.urllib.request.urlopen",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    assert TritonServer._ready_model_names() == {"ready"}


def test_gpu_telemetry_writes_header_and_stops_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()
    monkeypatch.setattr(
        "triton_benchmark.triton_runtime.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr("triton_benchmark.triton_runtime.os.killpg", lambda *args: None)
    output = tmp_path / "gpu.csv"

    with GpuTelemetry(output):
        pass

    assert output.read_text(encoding="utf-8").startswith("timestamp,utilization")
    assert process.waited is True


def test_process_telemetry_writes_server_cpu_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "triton_benchmark.triton_runtime._read_process_sample",
        lambda pid: (12.5, 2048),
    )
    output = tmp_path / "cpu.csv"

    with ProcessTelemetry(output, process_id=123, interval_seconds=0.01):
        pass

    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "timestamp_utc,process_cpu_seconds,resident_memory_kib"
    assert rows[1].endswith(",12.5,2048")


def test_grpc_adapter_sends_expected_tensor_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class InferInput:
        def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
            observed["input"] = (name, shape, dtype)

        def set_data_from_numpy(self, value: np.ndarray) -> None:
            observed["value"] = value

    class Client:
        def __init__(self, **kwargs: object) -> None:
            observed["client"] = kwargs

        async def infer(self, model_name: str, **kwargs: object) -> object:
            observed["model"] = model_name
            observed["request"] = kwargs
            return SimpleNamespace(as_numpy=lambda name: np.asarray([1.0]))

        async def close(self) -> None:
            observed["closed"] = True

    tritonclient = ModuleType("tritonclient")
    grpc = ModuleType("tritonclient.grpc")
    grpc.InferInput = InferInput  # type: ignore[attr-defined]
    grpc.InferRequestedOutput = lambda name: name  # type: ignore[attr-defined]
    grpc_aio = ModuleType("tritonclient.grpc.aio")
    grpc_aio.InferenceServerClient = Client  # type: ignore[attr-defined]
    grpc.aio = grpc_aio  # type: ignore[attr-defined]
    tritonclient.grpc = grpc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tritonclient", tritonclient)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc.aio", grpc_aio)

    async def exercise() -> None:
        async with TritonGrpcAdapter(backend="onnx") as adapter:
            await adapter("onnx_model_000", np.ones((1, 32), dtype=np.float32))

    asyncio.run(exercise())

    assert observed["input"] == ("input", (1, 32), "FP32")
    assert observed["model"] == "onnx_model_000"
    assert observed["closed"] is True


def test_execute_matrix_writes_complete_artifacts_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "models"
    results = tmp_path / "results"
    generate_repository(repository, count=1, seed=7)
    spec = MatrixRunSpec(
        backend="onnx",
        model_counts=(1,),
        queue_delays_microseconds=(None,),
        target_qps_values=(50.0,),
        repetitions=1,
        warmup_seconds=0.01,
        duration_seconds=0.02,
        base_seed=7,
        server_cpus="0",
        loadgen_cpus="1",
        image_digest="ghcr.io/example/image@sha256:" + "a" * 64,
        git_commit="abc1234",
    )

    class Server:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "Server":
            return self

        @property
        def pid(self) -> int:
            return 123

        def __exit__(self, *args: object) -> None:
            return None

    class Adapter:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "Adapter":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class Telemetry:
        def __init__(self, output_path: Path, **kwargs: object) -> None:
            self.output_path = output_path

        def __enter__(self) -> "Telemetry":
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("timestamp\n", encoding="utf-8")
            return self

        def __exit__(self, *args: object) -> None:
            return None

    async def fake_run_load(spec: object, **kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output_csv"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("request_id\n", encoding="utf-8")
        return {
            "measurement_requests": 10,
            "failed_requests": 0,
            "target_qps": 50.0,
            "achieved_qps": 50.0,
            "scheduler_delay_p95_ms": 1.0,
        }

    monkeypatch.setattr(
        "triton_benchmark.matrix_runner.validate_process_affinity", lambda value: 1
    )
    monkeypatch.setattr("triton_benchmark.matrix_runner.TritonServer", Server)
    monkeypatch.setattr("triton_benchmark.matrix_runner.TritonGrpcAdapter", Adapter)
    monkeypatch.setattr("triton_benchmark.matrix_runner.GpuTelemetry", Telemetry)
    monkeypatch.setattr("triton_benchmark.matrix_runner.ProcessTelemetry", Telemetry)
    monkeypatch.setattr("triton_benchmark.matrix_runner.run_load", fake_run_load)

    assert execute_matrix(
        spec, repository=repository, results_dir=results, resume=False
    ) == {"completed": 1, "skipped": 0, "total": 1}
    assert execute_matrix(
        spec, repository=repository, results_dir=results, resume=True
    ) == {"completed": 0, "skipped": 1, "total": 1}
    assert model_names("onnx", 1) == ("onnx_model_000",)
