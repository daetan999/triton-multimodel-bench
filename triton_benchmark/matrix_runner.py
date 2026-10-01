"""Durable orchestration for the reproducible Triton benchmark matrix."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator

from triton_benchmark.benchmark import (
    Backend,
    BenchmarkConfigurationError,
    LoadSpec,
    MatrixCondition,
    build_matrix,
    configure_batching,
    run_load,
)
from triton_benchmark.triton_runtime import (
    GpuTelemetry,
    ProcessTelemetry,
    TritonGrpcAdapter,
    TritonServer,
    build_triton_command,
    model_names,
)


@dataclass(frozen=True)
class MatrixRunSpec:
    """Every setting that must remain fixed when a matrix is resumed."""

    backend: Backend
    model_counts: tuple[int, ...]
    queue_delays_microseconds: tuple[int | None, ...]
    target_qps_values: tuple[float, ...]
    repetitions: int
    warmup_seconds: float
    duration_seconds: float
    base_seed: int
    server_cpus: str
    loadgen_cpus: str
    image_digest: str
    git_commit: str

    def conditions(self) -> tuple[MatrixCondition, ...]:
        return build_matrix(
            backend=self.backend,
            model_counts=self.model_counts,
            queue_delays_microseconds=self.queue_delays_microseconds,
            target_qps_values=self.target_qps_values,
            repetitions=self.repetitions,
            base_seed=self.base_seed,
        )

    def manifest(self) -> dict[str, Any]:
        document = asdict(self)
        document["schema_version"] = 1
        document["run_order"] = [condition.run_id for condition in self.conditions()]
        return json.loads(json.dumps(document))


@dataclass(frozen=True)
class ResultPaths:
    raw: Path
    gpu: Path
    cpu: Path
    summary: Path


class ResultStore:
    """Own the immutable manifest and resumable artifacts for one matrix."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "matrix-manifest.json"

    def initialize(self, spec: MatrixRunSpec, *, resume: bool) -> None:
        expected = spec.manifest()
        if self.manifest_path.exists():
            if not resume:
                raise BenchmarkConfigurationError(
                    f"results already exist at {self.root}; pass --resume only "
                    "to continue the exact same matrix"
                )
            observed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if observed != expected:
                raise BenchmarkConfigurationError(
                    "existing matrix manifest does not match requested settings"
                )
            return
        if resume:
            raise BenchmarkConfigurationError(
                f"cannot resume because {self.manifest_path} does not exist"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.manifest_path, expected)

    def paths_for(self, condition: MatrixCondition) -> ResultPaths:
        return ResultPaths(
            raw=self.root / "raw" / f"{condition.run_id}.csv",
            gpu=self.root / "gpu" / f"{condition.run_id}.csv",
            cpu=self.root / "cpu" / f"{condition.run_id}.csv",
            summary=self.root / "summaries" / f"{condition.run_id}.json",
        )

    def is_complete(self, condition: MatrixCondition) -> bool:
        paths = self.paths_for(condition)
        exists = tuple(path.exists() for path in asdict(paths).values())
        if not any(exists):
            return False
        if not all(exists):
            raise BenchmarkConfigurationError(
                f"partial artifacts found for {condition.run_id}; preserve them "
                "for diagnosis and use a new results directory"
            )
        summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        if summary.get("run_id") != condition.run_id:
            raise BenchmarkConfigurationError(
                f"summary run_id mismatch for {condition.run_id}"
            )
        if summary.get("status") != "valid":
            raise BenchmarkConfigurationError(
                f"completed artifacts for {condition.run_id} are not valid"
            )
        return True

    def write_summary(
        self, condition: MatrixCondition, summary: dict[str, Any]
    ) -> None:
        paths = self.paths_for(condition)
        if paths.summary.exists():
            raise BenchmarkConfigurationError(f"refusing to overwrite {paths.summary}")
        write_json_atomic(paths.summary, summary)


def condition_groups(
    conditions: Iterable[MatrixCondition],
) -> Iterator[tuple[tuple[int, int | None], tuple[MatrixCondition, ...]]]:
    """Yield restart groups that share model count and batching policy."""

    def key(condition: MatrixCondition) -> tuple[int, int | None]:
        return condition.model_count, condition.queue_delay_microseconds

    for group_key, members in groupby(conditions, key=key):
        yield group_key, tuple(members)


def assess_run(summary: dict[str, Any]) -> tuple[str, list[str]]:
    """Separate invalid runs from real, but possibly saturated, measurements."""
    warnings: list[str] = []
    if summary.get("measurement_requests", 0) < 1:
        return "invalid", ["no measured requests completed"]
    if summary.get("failed_requests", 0) != 0:
        return "invalid", ["one or more measured requests failed"]

    target_qps = float(summary["target_qps"])
    achieved_qps = float(summary["achieved_qps"])
    if achieved_qps < target_qps * 0.95:
        warnings.append("achieved QPS is below 95% of target")
    scheduler_delay = summary.get("scheduler_delay_p95_ms")
    if scheduler_delay is not None and float(scheduler_delay) > 100.0:
        warnings.append("client scheduler delay p95 exceeds 100 ms")
    return "valid", warnings


def execute_matrix(
    spec: MatrixRunSpec,
    *,
    repository: Path | str,
    results_dir: Path | str,
    resume: bool,
) -> dict[str, int]:
    """Run or resume a full matrix, restarting Triton only between groups."""
    validate_matrix_spec(spec)
    allocated_cpu_count = validate_process_affinity(spec.loadgen_cpus)
    repository_path = Path(repository).resolve()
    store = ResultStore(results_dir)
    store.initialize(spec, resume=resume)

    completed = 0
    skipped = 0
    conditions = spec.conditions()
    for (model_count, queue_delay), group in condition_groups(conditions):
        incomplete: list[MatrixCondition] = []
        for condition in group:
            if store.is_complete(condition):
                skipped += 1
            else:
                incomplete.append(condition)
        if not incomplete:
            print(
                f"group m={model_count} delay={queue_delay}: all runs complete",
                flush=True,
            )
            continue

        print(
            f"starting group m={model_count} delay={queue_delay} "
            f"pending={len(incomplete)}",
            flush=True,
        )
        configure_batching(
            repository_path,
            backend=spec.backend,
            queue_delay_microseconds=queue_delay,
        )
        delay_label = "off" if queue_delay is None else f"{queue_delay}us"
        server_log = (
            store.root
            / "server"
            / f"{spec.backend}_m{model_count:03d}_batch{delay_label}.log"
        )
        command = build_triton_command(
            repository=repository_path,
            backend=spec.backend,
            model_count=model_count,
            server_cpus=spec.server_cpus,
        )
        with TritonServer(
            command=command,
            expected_models=model_names(spec.backend, model_count),
            log_path=server_log,
        ) as server:
            asyncio.run(
                _execute_group(
                    spec,
                    incomplete,
                    store=store,
                    allocated_cpu_count=allocated_cpu_count,
                    server_log=server_log,
                    server_process_id=server.pid,
                )
            )
        completed += len(incomplete)
    return {"completed": completed, "skipped": skipped, "total": len(conditions)}


async def _execute_group(
    spec: MatrixRunSpec,
    conditions: Iterable[MatrixCondition],
    *,
    store: ResultStore,
    allocated_cpu_count: int,
    server_log: Path,
    server_process_id: int,
) -> None:
    async with TritonGrpcAdapter(backend=spec.backend) as infer:
        for condition in conditions:
            paths = store.paths_for(condition)
            print(f"starting run {condition.run_id}", flush=True)
            started_at = datetime.now(timezone.utc)
            with (
                GpuTelemetry(paths.gpu),
                ProcessTelemetry(paths.cpu, process_id=server_process_id),
            ):
                summary = await run_load(
                    LoadSpec(
                        backend=spec.backend,
                        model_count=condition.model_count,
                        target_qps=condition.target_qps,
                        warmup_seconds=spec.warmup_seconds,
                        duration_seconds=spec.duration_seconds,
                        seed=condition.seed,
                    ),
                    infer=infer,
                    output_csv=paths.raw,
                    allocated_cpu_count=allocated_cpu_count,
                )
            status, warnings = assess_run(summary)
            finished_at = datetime.now(timezone.utc)
            document = {
                **summary,
                "run_id": condition.run_id,
                "status": status,
                "warnings": warnings,
                "queue_delay_microseconds": condition.queue_delay_microseconds,
                "repetition": condition.repetition,
                "image_digest": spec.image_digest,
                "git_commit": spec.git_commit,
                "server_cpus": spec.server_cpus,
                "loadgen_cpus": spec.loadgen_cpus,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "artifacts": {
                    "raw": str(paths.raw),
                    "gpu": str(paths.gpu),
                    "cpu": str(paths.cpu),
                    "server_log": str(server_log),
                },
            }
            store.write_summary(condition, document)
            print(
                f"completed run {condition.run_id} status={status} "
                f"achieved_qps={summary['achieved_qps']:.3f}",
                flush=True,
            )
            if status != "valid":
                raise RuntimeError(
                    f"run {condition.run_id} is invalid: {', '.join(warnings)}"
                )


def parse_cpu_list(value: str) -> frozenset[int]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""
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
        raise BenchmarkConfigurationError(f"invalid CPU list: {value!r}") from error
    if not cpus:
        raise BenchmarkConfigurationError("CPU list must not be empty")
    return frozenset(cpus)


def validate_matrix_spec(spec: MatrixRunSpec) -> None:
    """Reject provenance or CPU settings that would weaken the experiment."""
    server_cpus = parse_cpu_list(spec.server_cpus)
    loadgen_cpus = parse_cpu_list(spec.loadgen_cpus)
    if server_cpus & loadgen_cpus:
        raise BenchmarkConfigurationError(
            "server and load-generator CPU sets must be disjoint"
        )
    if re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", spec.image_digest) is None:
        raise BenchmarkConfigurationError(
            "image_digest must be an immutable image digest ending in "
            "@sha256:<64 lowercase hex characters>"
        )
    if re.fullmatch(r"[0-9a-f]{7,40}", spec.git_commit) is None:
        raise BenchmarkConfigurationError(
            "git_commit must contain 7 to 40 lowercase hexadecimal characters"
        )


def validate_process_affinity(
    loadgen_cpus: str, *, current: Iterable[int] | None = None
) -> int:
    """Require the runner itself to be pinned to the reserved client CPUs."""
    expected = parse_cpu_list(loadgen_cpus)
    observed = frozenset(os.sched_getaffinity(0) if current is None else current)
    if observed != expected:
        raise BenchmarkConfigurationError(
            f"load generator affinity is {sorted(observed)}, expected "
            f"{sorted(expected)}; invoke with taskset --cpu-list {loadgen_cpus}"
        )
    return len(expected)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
