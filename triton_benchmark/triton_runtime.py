"""Triton process and gRPC adapters used by the benchmark runner."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from triton_benchmark.benchmark import Backend, BenchmarkConfigurationError


def model_names(backend: Backend, model_count: int) -> tuple[str, ...]:
    """Return the canonical generated model names for one backend."""
    if backend not in ("onnx", "fil"):
        raise BenchmarkConfigurationError(f"unsupported backend: {backend}")
    if model_count < 1:
        raise BenchmarkConfigurationError("model_count must be at least 1")
    prefix = "onnx_model" if backend == "onnx" else "fil_model"
    return tuple(f"{prefix}_{index:03d}" for index in range(model_count))


def build_triton_command(
    *,
    repository: Path | str,
    backend: Backend,
    model_count: int,
    server_cpus: str,
) -> tuple[str, ...]:
    """Build a CPU-pinned Triton command with an explicit model allowlist."""
    if not server_cpus.strip():
        raise BenchmarkConfigurationError("server_cpus must not be empty")
    repository_path = Path(repository).resolve()
    command = [
        "taskset",
        "--cpu-list",
        server_cpus,
        "/opt/tritonserver/bin/tritonserver",
        f"--model-repository={repository_path}",
        "--model-control-mode=explicit",
        "--model-load-thread-count=4",
        "--strict-model-config=true",
        "--http-port=8000",
        "--grpc-port=8001",
        "--metrics-port=8002",
    ]
    if backend == "onnx":
        command.extend(
            (
                "--backend-config=onnxruntime,enable-global-threadpool=1",
                "--backend-config=onnxruntime,"
                f"intra_op_thread_count={_cpu_count(server_cpus)}",
                "--backend-config=onnxruntime,inter_op_thread_count=1",
            )
        )
    command.extend(f"--load-model={name}" for name in model_names(backend, model_count))
    return tuple(command)


def _cpu_count(value: str) -> int:
    cpus: set[int] = set()
    try:
        for part in value.split(","):
            bounds = part.strip().split("-", maxsplit=1)
            start = int(bounds[0])
            end = int(bounds[-1])
            if start < 0 or end < start:
                raise ValueError
            cpus.update(range(start, end + 1))
    except (ValueError, IndexError) as error:
        raise BenchmarkConfigurationError(
            f"invalid server CPU list: {value!r}"
        ) from error
    if not cpus:
        raise BenchmarkConfigurationError("server CPU list must not be empty")
    return len(cpus)


class TritonServer:
    """Own one Triton process and prove its requested models are ready."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        expected_models: tuple[str, ...],
        log_path: Path | str,
        readiness_timeout_seconds: float = 180.0,
    ) -> None:
        self._command = command
        self._expected_models = expected_models
        self._log_path = Path(log_path)
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None

    def __enter__(self) -> "TritonServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Triton process is already started")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("ab")
        self._log_file.write(
            f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n".encode()
        )
        self._log_file.flush()
        self._process = subprocess.Popen(
            self._command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

    @property
    def pid(self) -> int:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Triton process is not running")
        return self._process.pid

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=10)
        self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._readiness_timeout_seconds
        last_error = "Triton did not answer"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Triton exited with {self._process.returncode}:\n{self._tail_log()}"
                )
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8000/v2/health/ready", timeout=2
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"readiness returned {response.status}")
                ready_models = self._ready_model_names()
                if set(self._expected_models) == ready_models:
                    return
                last_error = (
                    f"expected {len(self._expected_models)} ready models, "
                    f"found {len(ready_models)}"
                )
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = str(error)
            time.sleep(1)
        raise RuntimeError(
            f"Triton readiness timeout: {last_error}\n{self._tail_log()}"
        )

    @staticmethod
    def _ready_model_names() -> set[str]:
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v2/repository/index",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
        return {
            item["name"]
            for item in payload
            if item.get("state") == "READY" and item.get("version") == "1"
        }

    def _tail_log(self, lines: int = 80) -> str:
        if not self._log_path.is_file():
            return "log file was not created"
        return "\n".join(
            self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -lines:
            ]
        )


class GpuTelemetry:
    """Record one-second GPU utilization samples for exactly one load run."""

    def __init__(self, output_path: Path | str) -> None:
        self._output_path = Path(output_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._output_file: Any = None

    def __enter__(self) -> "GpuTelemetry":
        if self._output_path.exists():
            raise BenchmarkConfigurationError(
                f"refusing to overwrite {self._output_path}"
            )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_file = self._output_path.open("wb")
        self._output_file.write(
            b"timestamp,utilization_gpu_percent,memory_used_mib,power_draw_watts\n"
        )
        self._output_file.flush()
        self._process = subprocess.Popen(
            (
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "--loop=1",
            ),
            stdout=self._output_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._process is not None and self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=5)
        if self._output_file is not None:
            self._output_file.close()


class ProcessTelemetry:
    """Sample Triton process CPU time and resident memory once per interval."""

    def __init__(
        self,
        output_path: Path | str,
        *,
        process_id: int,
        interval_seconds: float = 1.0,
    ) -> None:
        self._output_path = Path(output_path)
        self._process_id = process_id
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_file: Any = None
        self._error: Exception | None = None

    def __enter__(self) -> "ProcessTelemetry":
        if self._output_path.exists():
            raise BenchmarkConfigurationError(
                f"refusing to overwrite {self._output_path}"
            )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_file = self._output_path.open("w", encoding="utf-8")
        self._output_file.write(
            "timestamp_utc,process_cpu_seconds,resident_memory_kib\n"
        )
        self._sample_once()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 2)
        if self._output_file is not None:
            self._output_file.close()
        if exc_type is None and self._error is not None:
            raise RuntimeError("Triton process telemetry failed") from self._error

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._sample_once()
            except Exception as error:
                self._error = error
                return

    def _sample_once(self) -> None:
        process_cpu_seconds, resident_memory_kib = _read_process_sample(
            self._process_id
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        self._output_file.write(
            f"{timestamp},{process_cpu_seconds},{resident_memory_kib}\n"
        )
        self._output_file.flush()


def _read_process_sample(process_id: int) -> tuple[float, int]:
    stat = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8")
    fields = stat.rsplit(")", maxsplit=1)[1].strip().split()
    clock_ticks = os.sysconf("SC_CLK_TCK")
    process_cpu_seconds = (int(fields[11]) + int(fields[12])) / clock_ticks
    status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    resident_memory_kib = next(
        int(line.split()[1])
        for line in status.splitlines()
        if line.startswith("VmRSS:")
    )
    return process_cpu_seconds, resident_memory_kib


class TritonGrpcAdapter:
    """Translate benchmark requests into Triton gRPC inference calls."""

    def __init__(self, *, backend: Backend, url: str = "127.0.0.1:8001") -> None:
        if backend not in ("onnx", "fil"):
            raise BenchmarkConfigurationError(f"unsupported backend: {backend}")
        try:
            import tritonclient.grpc as grpc
            import tritonclient.grpc.aio as grpc_aio
        except ImportError as error:
            raise RuntimeError(
                "tritonclient[grpc] is required for live benchmark runs"
            ) from error
        self._backend = backend
        self._grpc = grpc
        self._client = grpc_aio.InferenceServerClient(url=url, verbose=False)

    async def __aenter__(self) -> "TritonGrpcAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._client.close()

    async def __call__(self, model_name: str, features: np.ndarray) -> None:
        input_name, output_name = (
            ("input", "variable")
            if self._backend == "onnx"
            else ("input__0", "output__0")
        )
        request_input = self._grpc.InferInput(input_name, features.shape, "FP32")
        request_input.set_data_from_numpy(features)
        response = await self._client.infer(
            model_name,
            inputs=[request_input],
            outputs=[self._grpc.InferRequestedOutput(output_name)],
        )
        output = response.as_numpy(output_name)
        if output is None or output.size != features.shape[0]:
            raise RuntimeError(
                f"unexpected {output_name} output from {model_name}: "
                f"{None if output is None else output.shape}"
            )
