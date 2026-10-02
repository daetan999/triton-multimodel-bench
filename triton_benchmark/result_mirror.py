"""Continuously mirror benchmark artifacts from a disposable RunPod disk."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class MirrorConfiguration:
    """Connection and path settings for one result mirror."""

    host: str
    port: int
    identity_file: Path
    known_hosts_file: Path
    remote_dir: str
    local_dir: Path


def build_rsync_command(config: MirrorConfiguration) -> tuple[str, ...]:
    """Build a strict, non-destructive rsync command."""
    _validate_configuration(config)
    ssh_command = shlex.join(
        (
            "ssh",
            "-i",
            str(config.identity_file),
            "-p",
            str(config.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={config.known_hosts_file}",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
        )
    )
    remote_dir = config.remote_dir.rstrip("/") + "/"
    local_dir = str(config.local_dir) + "/"
    return (
        "rsync",
        "--archive",
        "--partial",
        "--safe-links",
        "-e",
        ssh_command,
        f"root@{config.host}:{remote_dir}",
        local_dir,
    )


def mirror_once(config: MirrorConfiguration) -> None:
    """Copy all currently available artifacts without removing local files."""
    command = build_rsync_command(config)
    config.local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def _validate_configuration(config: MirrorConfiguration) -> None:
    if not re.fullmatch(r"[A-Za-z0-9.-]+", config.host):
        raise ValueError("host must be an IPv4 address or DNS hostname")
    if not 1 <= config.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")

    remote_path = PurePosixPath(config.remote_dir)
    allowed_root = PurePosixPath("/workspace/results")
    if not remote_path.is_absolute() or allowed_root not in remote_path.parents:
        raise ValueError("remote directory must be below /workspace/results")
    relative_parts = remote_path.relative_to(allowed_root).parts
    if any(
        part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", part)
        for part in relative_parts
    ):
        raise ValueError("remote directory below /workspace/results is unsafe")
