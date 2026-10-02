#!/usr/bin/env python3
"""Continuously copy benchmark results from a RunPod to this computer."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_benchmark.result_mirror import MirrorConfiguration, mirror_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--identity-file", required=True, type=Path)
    parser.add_argument("--known-hosts-file", required=True, type=Path)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--interval-seconds", default=30.0, type=float)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")

    config = MirrorConfiguration(
        host=args.host,
        port=args.port,
        identity_file=args.identity_file,
        known_hosts_file=args.known_hosts_file,
        remote_dir=args.remote_dir,
        local_dir=args.local_dir,
    )
    while True:
        try:
            mirror_once(config)
            print("result mirror completed", flush=True)
        except subprocess.CalledProcessError as error:
            print(f"result mirror failed: {error}", file=sys.stderr, flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
