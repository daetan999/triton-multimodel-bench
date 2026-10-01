#!/usr/bin/env python3
"""Validate the paired Triton model repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_benchmark.model_zoo import validate_repository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("model_repository"),
    )
    args = parser.parse_args()

    report = validate_repository(args.repository, expected_count=args.count)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
