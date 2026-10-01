#!/usr/bin/env python3
"""Generate the paired Triton model repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_benchmark.model_zoo import generate_repository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("model_repository"),
    )
    args = parser.parse_args()

    manifest = generate_repository(
        args.repository,
        count=args.count,
        seed=args.seed,
    )
    print(
        f"Generated {manifest['count']} ONNX/FIL model pairs in "
        f"{args.repository.resolve()}"
    )


if __name__ == "__main__":
    main()
