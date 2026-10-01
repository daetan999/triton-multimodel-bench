#!/usr/bin/env python3
"""Run or safely resume the formal Triton benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_benchmark.matrix_runner import MatrixRunSpec, execute_matrix


def comma_separated_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def comma_separated_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def queue_delays(value: str) -> tuple[int | None, ...]:
    return tuple(
        None if item.lower() == "off" else int(item) for item in value.split(",")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnx", "fil"), default="onnx")
    parser.add_argument(
        "--model-counts", type=comma_separated_ints, default=(1, 10, 100)
    )
    parser.add_argument(
        "--queue-delays-us", type=queue_delays, default=(None, 2_000, 10_000)
    )
    parser.add_argument(
        "--target-qps", type=comma_separated_floats, default=(50, 200, 500)
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--base-seed", type=int, default=20260805)
    parser.add_argument("--server-cpus", default="0-7")
    parser.add_argument("--loadgen-cpus", default="8-15")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    report = execute_matrix(
        MatrixRunSpec(
            backend=args.backend,
            model_counts=args.model_counts,
            queue_delays_microseconds=args.queue_delays_us,
            target_qps_values=args.target_qps,
            repetitions=args.repetitions,
            warmup_seconds=args.warmup_seconds,
            duration_seconds=args.duration_seconds,
            base_seed=args.base_seed,
            server_cpus=args.server_cpus,
            loadgen_cpus=args.loadgen_cpus,
            image_digest=args.image_digest,
            git_commit=args.git_commit,
        ),
        repository=args.repository,
        results_dir=args.results_dir,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
