"""Benchmark configuration, traffic generation, and result helpers."""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Sequence

import numpy as np


Backend = Literal["onnx", "fil"]
Phase = Literal["warmup", "measure"]

_BATCHING_START = "# BEGIN managed dynamic batching"
_BATCHING_END = "# END managed dynamic batching"
_BATCHING_PATTERN = re.compile(
    rf"\n?{re.escape(_BATCHING_START)}.*?{re.escape(_BATCHING_END)}\n?",
    flags=re.DOTALL,
)


class BenchmarkConfigurationError(ValueError):
    """Raised when a benchmark configuration is invalid or ambiguous."""


@dataclass(frozen=True)
class PlannedRequest:
    """One deterministic request in an open-loop arrival plan."""

    request_id: int
    scheduled_offset_seconds: float
    model_name: str
    phase: Phase
    input_seed: int


@dataclass(frozen=True)
class RequestRecord:
    """Observed outcome for one scheduled inference request."""

    request_id: int
    model_name: str
    phase: Phase
    scheduled_offset_seconds: float
    started_offset_seconds: float
    completed_offset_seconds: float
    latency_ms: float
    success: bool
    error: str


@dataclass(frozen=True)
class LoadSpec:
    """Configuration for one open-loop benchmark run."""

    backend: Backend
    model_count: int
    target_qps: float
    warmup_seconds: float
    duration_seconds: float
    seed: int


@dataclass(frozen=True)
class MatrixCondition:
    """One uniquely named condition in a reproducible benchmark matrix."""

    backend: Backend
    model_count: int
    queue_delay_microseconds: int | None
    target_qps: float
    repetition: int
    seed: int

    @property
    def run_id(self) -> str:
        if self.queue_delay_microseconds is None:
            delay = "off"
        elif self.queue_delay_microseconds % 1_000 == 0:
            delay = f"{self.queue_delay_microseconds // 1_000}ms"
        else:
            delay = f"{self.queue_delay_microseconds}us"
        qps = format(self.target_qps, "g").replace(".", "p")
        return (
            f"{self.backend}_m{self.model_count:03d}_batch{delay}"
            f"_qps{qps}_r{self.repetition:02d}"
        )


def build_matrix(
    *,
    backend: Backend,
    model_counts: Sequence[int],
    queue_delays_microseconds: Sequence[int | None],
    target_qps_values: Sequence[float],
    repetitions: int,
    base_seed: int,
) -> tuple[MatrixCondition, ...]:
    """Build a complete matrix with randomized group and run order."""
    if backend not in ("onnx", "fil"):
        raise BenchmarkConfigurationError(f"unsupported backend: {backend}")
    if repetitions < 1:
        raise BenchmarkConfigurationError("repetitions must be at least 1")
    if base_seed < 0:
        raise BenchmarkConfigurationError("base_seed must be non-negative")
    if not model_counts or any(count < 1 for count in model_counts):
        raise BenchmarkConfigurationError("model counts must be positive")
    if not target_qps_values or any(qps <= 0 for qps in target_qps_values):
        raise BenchmarkConfigurationError("target QPS values must be positive")
    if not queue_delays_microseconds or any(
        delay is not None and delay <= 0 for delay in queue_delays_microseconds
    ):
        raise BenchmarkConfigurationError("queue delays must be positive or None")
    for label, values in (
        ("model counts", model_counts),
        ("queue delays", queue_delays_microseconds),
        ("target QPS values", target_qps_values),
    ):
        if len(set(values)) != len(values):
            raise BenchmarkConfigurationError(f"{label} must not contain duplicates")

    groups: dict[tuple[int, int | None], list[MatrixCondition]] = {}
    seed_offset = 0
    for model_count in model_counts:
        for queue_delay in queue_delays_microseconds:
            group: list[MatrixCondition] = []
            for target_qps in target_qps_values:
                for repetition in range(1, repetitions + 1):
                    group.append(
                        MatrixCondition(
                            backend=backend,
                            model_count=model_count,
                            queue_delay_microseconds=queue_delay,
                            target_qps=float(target_qps),
                            repetition=repetition,
                            seed=base_seed + seed_offset,
                        )
                    )
                    seed_offset += 1
            groups[(model_count, queue_delay)] = group

    order_rng = np.random.default_rng(base_seed ^ 0xB47C)
    group_keys = list(groups)
    order_rng.shuffle(group_keys)
    ordered: list[MatrixCondition] = []
    for key in group_keys:
        group = groups[key]
        order_rng.shuffle(group)
        ordered.extend(group)
    return tuple(ordered)


InferenceAdapter = Callable[[str, np.ndarray], Awaitable[None]]


async def run_load(
    spec: LoadSpec,
    *,
    infer: InferenceAdapter,
    output_csv: Path | str,
    allocated_cpu_count: int,
) -> dict[str, Any]:
    """Execute one open-loop run and atomically write every request outcome."""
    output_path = Path(output_csv)
    if output_path.exists():
        raise BenchmarkConfigurationError(f"refusing to overwrite {output_path}")
    plan = build_arrival_plan(
        backend=spec.backend,
        model_count=spec.model_count,
        target_qps=spec.target_qps,
        warmup_seconds=spec.warmup_seconds,
        duration_seconds=spec.duration_seconds,
        seed=spec.seed,
    )
    if not plan:
        raise BenchmarkConfigurationError("arrival plan contains no requests")

    started_at = time.perf_counter()
    cpu_started_at = time.process_time()
    in_flight = 0
    max_in_flight = 0

    async def execute(planned: PlannedRequest) -> RequestRecord:
        nonlocal in_flight, max_in_flight
        features = (
            np.random.default_rng(planned.input_seed)
            .normal(size=(1, 32))
            .astype(np.float32)
        )
        request_started = time.perf_counter()
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        success = True
        error_message = ""
        try:
            await infer(planned.model_name, features)
        except Exception as error:  # The raw data must retain request failures.
            success = False
            error_message = f"{type(error).__name__}: {error}"[:500]
        finally:
            in_flight -= 1
        completed_at = time.perf_counter()
        return RequestRecord(
            request_id=planned.request_id,
            model_name=planned.model_name,
            phase=planned.phase,
            scheduled_offset_seconds=planned.scheduled_offset_seconds,
            started_offset_seconds=request_started - started_at,
            completed_offset_seconds=completed_at - started_at,
            latency_ms=(completed_at - request_started) * 1000.0,
            success=success,
            error=error_message,
        )

    tasks: list[asyncio.Task[RequestRecord]] = []
    for planned in plan:
        delay = started_at + planned.scheduled_offset_seconds - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(execute(planned)))
    records = sorted(await asyncio.gather(*tasks), key=lambda record: record.request_id)
    process_cpu_seconds = time.process_time() - cpu_started_at
    measurement_completions = [
        record.completed_offset_seconds - spec.warmup_seconds
        for record in records
        if record.phase == "measure"
    ]
    measurement_elapsed = max(
        spec.duration_seconds,
        max(measurement_completions, default=spec.duration_seconds),
    )

    _write_records(output_path, records)
    summary = summarize_records(
        records,
        target_qps=spec.target_qps,
        measurement_elapsed_seconds=measurement_elapsed,
        process_cpu_seconds=process_cpu_seconds,
        allocated_cpu_count=allocated_cpu_count,
    )
    summary.update(
        {
            "backend": spec.backend,
            "model_count": spec.model_count,
            "seed": spec.seed,
            "warmup_seconds": spec.warmup_seconds,
            "duration_seconds": spec.duration_seconds,
            "warmup_requests": sum(record.phase == "warmup" for record in records),
            "total_requests": len(records),
            "max_in_flight": max_in_flight,
            "client_cpu_scope": "whole run including warmup",
        }
    )
    return summary


def _write_records(output_path: Path, records: Sequence[RequestRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    fieldnames = list(RequestRecord.__dataclass_fields__)
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fieldnames})
    temporary_path.replace(output_path)


def summarize_records(
    records: Sequence[RequestRecord],
    *,
    target_qps: float,
    measurement_elapsed_seconds: float,
    process_cpu_seconds: float,
    allocated_cpu_count: int,
) -> dict[str, Any]:
    """Summarize measured requests while retaining failures in the denominator."""
    if measurement_elapsed_seconds <= 0:
        raise BenchmarkConfigurationError(
            "measurement_elapsed_seconds must be positive"
        )
    if allocated_cpu_count < 1:
        raise BenchmarkConfigurationError("allocated_cpu_count must be at least 1")

    measured = [record for record in records if record.phase == "measure"]
    successful = [record for record in measured if record.success]
    failed_count = len(measured) - len(successful)
    latencies = np.asarray(
        [record.latency_ms for record in successful], dtype=np.float64
    )
    scheduler_delays = np.asarray(
        [
            max(
                0.0,
                (record.started_offset_seconds - record.scheduled_offset_seconds)
                * 1000.0,
            )
            for record in measured
        ],
        dtype=np.float64,
    )

    latency_summary: dict[str, float | None]
    if latencies.size:
        latency_summary = {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }
    else:
        latency_summary = {"mean": None, "p50": None, "p95": None, "p99": None}

    return {
        "target_qps": target_qps,
        "measurement_requests": len(measured),
        "successful_requests": len(successful),
        "failed_requests": failed_count,
        "error_rate": failed_count / len(measured) if measured else 0.0,
        "measurement_elapsed_seconds": measurement_elapsed_seconds,
        "achieved_qps": len(successful) / measurement_elapsed_seconds,
        "latency_ms": latency_summary,
        "scheduler_delay_p95_ms": (
            float(np.percentile(scheduler_delays, 95))
            if scheduler_delays.size
            else None
        ),
        "client_cpu_utilization_percent": (
            process_cpu_seconds
            / measurement_elapsed_seconds
            / allocated_cpu_count
            * 100.0
        ),
    }


def build_arrival_plan(
    *,
    backend: Backend,
    model_count: int,
    target_qps: float,
    warmup_seconds: float,
    duration_seconds: float,
    seed: int,
) -> tuple[PlannedRequest, ...]:
    """Build deterministic Poisson arrivals for warm-up and measurement."""
    if backend not in ("onnx", "fil"):
        raise BenchmarkConfigurationError(f"unsupported backend: {backend}")
    if model_count < 1:
        raise BenchmarkConfigurationError("model_count must be at least 1")
    if target_qps <= 0:
        raise BenchmarkConfigurationError("target_qps must be positive")
    if warmup_seconds < 0:
        raise BenchmarkConfigurationError("warmup_seconds must be non-negative")
    if duration_seconds <= 0:
        raise BenchmarkConfigurationError("duration_seconds must be positive")
    if seed < 0:
        raise BenchmarkConfigurationError("seed must be non-negative")

    arrival_seed, model_seed, input_seed = np.random.SeedSequence(seed).spawn(3)
    arrival_rng = np.random.default_rng(arrival_seed)
    model_rng = np.random.default_rng(model_seed)
    input_rng = np.random.default_rng(input_seed)
    total_seconds = warmup_seconds + duration_seconds
    prefix = "onnx_model" if backend == "onnx" else "fil_model"
    requests: list[PlannedRequest] = []
    scheduled_offset = 0.0

    while True:
        scheduled_offset += float(arrival_rng.exponential(1.0 / target_qps))
        if scheduled_offset >= total_seconds:
            break
        model_index = int(model_rng.integers(0, model_count))
        requests.append(
            PlannedRequest(
                request_id=len(requests),
                scheduled_offset_seconds=scheduled_offset,
                model_name=f"{prefix}_{model_index:03d}",
                phase="warmup" if scheduled_offset < warmup_seconds else "measure",
                input_seed=int(input_rng.integers(0, 2**32, dtype=np.uint64)),
            )
        )

    return tuple(requests)


def configure_batching(
    repository: Path | str,
    *,
    backend: Backend,
    queue_delay_microseconds: int | None,
) -> dict[str, int | str | None]:
    """Set one backend's generated configs to batching off or a fixed delay."""
    if backend not in ("onnx", "fil"):
        raise BenchmarkConfigurationError(f"unsupported backend: {backend}")
    if queue_delay_microseconds is not None and queue_delay_microseconds <= 0:
        raise BenchmarkConfigurationError(
            "queue_delay_microseconds must be positive or None"
        )

    repository_path = Path(repository)
    manifest_path = repository_path / "manifest.json"
    if not manifest_path.is_file():
        raise BenchmarkConfigurationError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("models")
    if not isinstance(entries, list) or not entries:
        raise BenchmarkConfigurationError("manifest contains no model entries")

    model_key = f"{backend}_model"
    configured = 0
    for entry in entries:
        model_name = entry.get(model_key)
        if not isinstance(model_name, str):
            raise BenchmarkConfigurationError(f"manifest entry is missing {model_key}")
        config_path = repository_path / model_name / "config.pbtxt"
        config = config_path.read_text(encoding="utf-8")
        config = _BATCHING_PATTERN.sub("\n", config).rstrip() + "\n"
        if "dynamic_batching" in config:
            raise BenchmarkConfigurationError(
                f"unmanaged dynamic_batching block in {config_path}"
            )
        if queue_delay_microseconds is not None:
            config += _batching_block(queue_delay_microseconds)
        config_path.write_text(config, encoding="utf-8")
        configured += 1

    return {
        "backend": backend,
        "models": configured,
        "queue_delay_microseconds": queue_delay_microseconds,
    }


def _batching_block(queue_delay_microseconds: int) -> str:
    return f"""{_BATCHING_START}
dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: {queue_delay_microseconds}
}}
{_BATCHING_END}
"""
