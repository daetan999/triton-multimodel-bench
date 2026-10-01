#!/usr/bin/env python3
"""Run one asynchronous open-loop gRPC load test against local Triton."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_benchmark.benchmark import LoadSpec, run_load
from triton_benchmark.matrix_runner import assess_run, write_json_atomic
from triton_benchmark.triton_runtime import TritonGrpcAdapter


def default_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


async def run(args: argparse.Namespace) -> dict[str, object]:
    async with TritonGrpcAdapter(backend=args.backend, url=args.triton_url) as infer:
        summary = await run_load(
            LoadSpec(
                backend=args.backend,
                model_count=args.model_count,
                target_qps=args.target_qps,
                warmup_seconds=args.warmup_seconds,
                duration_seconds=args.duration_seconds,
                seed=args.seed,
            ),
            infer=infer,
            output_csv=args.output_csv,
            allocated_cpu_count=args.allocated_cpu_count,
        )
    status, warnings = assess_run(summary)
    return {**summary, "status": status, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("onnx", "fil"), required=True)
    parser.add_argument("--model-count", type=int, required=True)
    parser.add_argument("--target-qps", type=float, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--triton-url", default="127.0.0.1:8001")
    parser.add_argument(
        "--allocated-cpu-count",
        type=int,
        default=default_cpu_count(),
    )
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    write_json_atomic(args.summary_json, summary)
    if summary["status"] != "valid":
        raise SystemExit("benchmark run was invalid; inspect the summary and raw CSV")


if __name__ == "__main__":
    main()
