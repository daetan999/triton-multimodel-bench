#!/usr/bin/env python3
"""Apply a managed Triton batching policy to generated model configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_benchmark.benchmark import configure_batching


def queue_delay(value: str) -> int | None:
    if value.lower() == "off":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("queue delay must be 'off' or positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("onnx", "fil"), required=True)
    parser.add_argument("--queue-delay-us", type=queue_delay, required=True)
    parser.add_argument("--repository", type=Path, default=Path("model_repository"))
    args = parser.parse_args()
    report = configure_batching(
        args.repository,
        backend=args.backend,
        queue_delay_microseconds=args.queue_delay_us,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
