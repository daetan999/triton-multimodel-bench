import asyncio
import csv
import json
from pathlib import Path

import numpy as np

from triton_benchmark.benchmark import (
    RequestRecord,
    LoadSpec,
    build_matrix,
    build_arrival_plan,
    configure_batching,
    run_load,
    summarize_records,
)
from triton_benchmark.model_zoo import generate_repository
from triton_benchmark.matrix_runner import (
    MatrixRunSpec,
    ResultStore,
    assess_run,
    condition_groups,
    validate_matrix_spec,
    validate_process_affinity,
)
from triton_benchmark.triton_runtime import build_triton_command


def test_configure_batching_updates_only_the_selected_backend(tmp_path: Path) -> None:
    repository = tmp_path / "models"
    generate_repository(repository, count=1, seed=20260805)
    fil_before = (repository / "fil_model_000" / "config.pbtxt").read_text(
        encoding="utf-8"
    )

    report = configure_batching(
        repository,
        backend="onnx",
        queue_delay_microseconds=10_000,
    )

    onnx_config = (repository / "onnx_model_000" / "config.pbtxt").read_text(
        encoding="utf-8"
    )
    assert report == {
        "backend": "onnx",
        "models": 1,
        "queue_delay_microseconds": 10_000,
    }
    assert "dynamic_batching {" in onnx_config
    assert "preferred_batch_size: [ 4, 8 ]" in onnx_config
    assert "max_queue_delay_microseconds: 10000" in onnx_config
    assert (repository / "fil_model_000" / "config.pbtxt").read_text(
        encoding="utf-8"
    ) == fil_before


def test_configure_batching_can_return_configs_to_batching_off(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "models"
    generate_repository(repository, count=1, seed=20260805)
    configure_batching(
        repository,
        backend="onnx",
        queue_delay_microseconds=2_000,
    )

    report = configure_batching(
        repository,
        backend="onnx",
        queue_delay_microseconds=None,
    )

    config = (repository / "onnx_model_000" / "config.pbtxt").read_text(
        encoding="utf-8"
    )
    assert report["queue_delay_microseconds"] is None
    assert "dynamic_batching" not in config
    assert "managed dynamic batching" not in config


def test_arrival_plan_is_reproducible_poisson_traffic_across_models() -> None:
    first = build_arrival_plan(
        backend="onnx",
        model_count=10,
        target_qps=100,
        warmup_seconds=0.2,
        duration_seconds=1.0,
        seed=42,
    )
    second = build_arrival_plan(
        backend="onnx",
        model_count=10,
        target_qps=100,
        warmup_seconds=0.2,
        duration_seconds=1.0,
        seed=42,
    )

    assert first == second
    assert first
    assert all(
        earlier.scheduled_offset_seconds < later.scheduled_offset_seconds
        for earlier, later in zip(first, first[1:])
    )
    assert {request.phase for request in first} == {"warmup", "measure"}
    assert all(request.model_name.startswith("onnx_model_00") for request in first)
    assert max(request.scheduled_offset_seconds for request in first) < 1.2


def test_summary_excludes_warmup_and_reports_failures() -> None:
    records = [
        RequestRecord(0, "onnx_model_000", "warmup", 0.1, 0.1, 0.101, 1.0, True, ""),
        RequestRecord(1, "onnx_model_000", "measure", 1.1, 1.1, 1.101, 1.0, True, ""),
        RequestRecord(2, "onnx_model_001", "measure", 1.2, 1.2, 1.202, 2.0, True, ""),
        RequestRecord(3, "onnx_model_002", "measure", 1.3, 1.3, 1.303, 3.0, True, ""),
        RequestRecord(4, "onnx_model_003", "measure", 1.4, 1.4, 1.404, 4.0, True, ""),
        RequestRecord(
            5, "onnx_model_004", "measure", 1.5, 1.5, 1.505, 5.0, False, "timeout"
        ),
    ]

    summary = summarize_records(
        records,
        target_qps=3.0,
        measurement_elapsed_seconds=2.0,
        process_cpu_seconds=1.0,
        allocated_cpu_count=4,
    )

    assert summary["measurement_requests"] == 5
    assert summary["successful_requests"] == 4
    assert summary["failed_requests"] == 1
    assert summary["error_rate"] == 0.2
    assert summary["achieved_qps"] == 2.0
    assert summary["latency_ms"]["p50"] == 2.5
    assert summary["client_cpu_utilization_percent"] == 12.5


def test_run_load_writes_every_request_and_excludes_warmup_from_summary(
    tmp_path: Path,
) -> None:
    observed_models: list[str] = []

    async def infer(model_name: str, features: np.ndarray) -> None:
        assert features.shape == (1, 32)
        assert features.dtype == np.float32
        observed_models.append(model_name)
        await asyncio.sleep(0)

    output = tmp_path / "raw.csv"
    summary = asyncio.run(
        run_load(
            LoadSpec(
                backend="onnx",
                model_count=3,
                target_qps=500,
                warmup_seconds=0.01,
                duration_seconds=0.03,
                seed=7,
            ),
            infer=infer,
            output_csv=output,
            allocated_cpu_count=2,
        )
    )

    with output.open(newline="", encoding="utf-8") as raw_file:
        rows = list(csv.DictReader(raw_file))
    assert len(rows) == len(observed_models)
    assert {row["phase"] for row in rows} == {"warmup", "measure"}
    assert summary["warmup_requests"] > 0
    assert summary["measurement_requests"] > 0
    assert summary["failed_requests"] == 0


def test_build_matrix_covers_every_condition_once_in_reproducible_order() -> None:
    first = build_matrix(
        backend="onnx",
        model_counts=(1, 10, 100),
        queue_delays_microseconds=(None, 2_000, 10_000),
        target_qps_values=(50, 200, 500),
        repetitions=3,
        base_seed=20260805,
    )
    second = build_matrix(
        backend="onnx",
        model_counts=(1, 10, 100),
        queue_delays_microseconds=(None, 2_000, 10_000),
        target_qps_values=(50, 200, 500),
        repetitions=3,
        base_seed=20260805,
    )

    assert first == second
    assert len(first) == 81
    assert len({condition.run_id for condition in first}) == 81
    assert {condition.model_count for condition in first} == {1, 10, 100}
    assert {condition.queue_delay_microseconds for condition in first} == {
        None,
        2_000,
        10_000,
    }
    assert {condition.target_qps for condition in first} == {50, 200, 500}
    assert {condition.repetition for condition in first} == {1, 2, 3}


def test_triton_command_loads_only_the_requested_backend_and_model_count(
    tmp_path: Path,
) -> None:
    command = build_triton_command(
        repository=tmp_path / "models",
        backend="fil",
        model_count=3,
        server_cpus="0-7",
    )

    assert command[:3] == (
        "taskset",
        "--cpu-list",
        "0-7",
    )
    assert "/opt/tritonserver/bin/tritonserver" in command
    assert f"--model-repository={tmp_path / 'models'}" in command
    assert "--model-control-mode=explicit" in command
    assert [
        argument for argument in command if argument.startswith("--load-model=")
    ] == [
        "--load-model=fil_model_000",
        "--load-model=fil_model_001",
        "--load-model=fil_model_002",
    ]


def test_result_store_resumes_only_complete_valid_runs(tmp_path: Path) -> None:
    condition = build_matrix(
        backend="onnx",
        model_counts=(1,),
        queue_delays_microseconds=(None,),
        target_qps_values=(50,),
        repetitions=1,
        base_seed=7,
    )[0]
    store = ResultStore(tmp_path)
    paths = store.paths_for(condition)

    assert store.is_complete(condition) is False
    paths.raw.parent.mkdir(parents=True)
    paths.raw.write_text("request_id\n", encoding="utf-8")
    try:
        store.is_complete(condition)
    except ValueError as error:
        assert "partial artifacts" in str(error)
    else:
        raise AssertionError("partial artifacts must not be treated as resumable")

    paths.gpu.parent.mkdir(parents=True)
    paths.gpu.write_text("timestamp\n", encoding="utf-8")
    paths.cpu.parent.mkdir(parents=True)
    paths.cpu.write_text("timestamp\n", encoding="utf-8")
    paths.summary.parent.mkdir(parents=True)
    paths.summary.write_text(
        json.dumps({"run_id": condition.run_id, "status": "valid"}),
        encoding="utf-8",
    )
    assert store.is_complete(condition) is True


def test_result_store_rejects_changed_manifest_on_resume(tmp_path: Path) -> None:
    first = MatrixRunSpec(
        backend="onnx",
        model_counts=(1, 10),
        queue_delays_microseconds=(None, 2_000),
        target_qps_values=(50.0,),
        repetitions=1,
        warmup_seconds=10.0,
        duration_seconds=60.0,
        base_seed=7,
        server_cpus="0-7",
        loadgen_cpus="8-15",
        image_digest="ghcr.io/example/image@sha256:abc",
        git_commit="abc123",
    )
    store = ResultStore(tmp_path)
    store.initialize(first, resume=False)
    store.initialize(first, resume=True)

    changed = MatrixRunSpec(**{**first.__dict__, "duration_seconds": 30.0})
    try:
        store.initialize(changed, resume=True)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("resume must reject a changed matrix")


def test_process_affinity_must_match_reserved_loadgen_cores() -> None:
    assert validate_process_affinity("8-10,12", current={8, 9, 10, 12}) == 4
    try:
        validate_process_affinity("8-15", current={8, 9})
    except ValueError as error:
        assert "taskset --cpu-list 8-15" in str(error)
    else:
        raise AssertionError("mismatched affinity must stop the benchmark")


def test_condition_groups_keep_each_server_configuration_contiguous() -> None:
    conditions = build_matrix(
        backend="onnx",
        model_counts=(1, 10),
        queue_delays_microseconds=(None, 2_000),
        target_qps_values=(50, 200),
        repetitions=2,
        base_seed=7,
    )
    groups = list(condition_groups(conditions))

    assert len(groups) == 4
    assert all(len(members) == 4 for _, members in groups)
    for key, members in groups:
        assert {
            (condition.model_count, condition.queue_delay_microseconds)
            for condition in members
        } == {key}


def test_assess_run_keeps_saturation_as_valid_but_rejects_failures() -> None:
    status, warnings = assess_run(
        {
            "measurement_requests": 100,
            "failed_requests": 0,
            "target_qps": 500,
            "achieved_qps": 400,
            "scheduler_delay_p95_ms": 150,
        }
    )
    assert status == "valid"
    assert len(warnings) == 2

    status, warnings = assess_run(
        {
            "measurement_requests": 100,
            "failed_requests": 1,
            "target_qps": 50,
            "achieved_qps": 49,
            "scheduler_delay_p95_ms": 1,
        }
    )
    assert status == "invalid"
    assert warnings == ["one or more measured requests failed"]


def test_matrix_spec_requires_disjoint_cpus_and_immutable_provenance() -> None:
    base = MatrixRunSpec(
        backend="onnx",
        model_counts=(1,),
        queue_delays_microseconds=(None,),
        target_qps_values=(50.0,),
        repetitions=1,
        warmup_seconds=10.0,
        duration_seconds=60.0,
        base_seed=7,
        server_cpus="0-7",
        loadgen_cpus="8-15",
        image_digest="ghcr.io/example/image@sha256:" + "a" * 64,
        git_commit="abc1234",
    )
    validate_matrix_spec(base)

    overlapping = MatrixRunSpec(**{**base.__dict__, "loadgen_cpus": "7-15"})
    try:
        validate_matrix_spec(overlapping)
    except ValueError as error:
        assert "must be disjoint" in str(error)
    else:
        raise AssertionError("overlapping CPU sets must be rejected")

    mutable_image = MatrixRunSpec(
        **{**base.__dict__, "image_digest": "ghcr.io/example/image:0.1.2"}
    )
    try:
        validate_matrix_spec(mutable_image)
    except ValueError as error:
        assert "immutable image digest" in str(error)
    else:
        raise AssertionError("mutable image references must be rejected")


def test_non_millisecond_queue_delays_have_unique_run_ids() -> None:
    conditions = build_matrix(
        backend="onnx",
        model_counts=(1,),
        queue_delays_microseconds=(2_000, 2_500),
        target_qps_values=(50,),
        repetitions=1,
        base_seed=7,
    )
    assert len({condition.run_id for condition in conditions}) == 2
