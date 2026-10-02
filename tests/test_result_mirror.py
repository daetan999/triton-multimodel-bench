import subprocess
from pathlib import Path

import pytest

from triton_benchmark.result_mirror import (
    MirrorConfiguration,
    build_rsync_command,
    mirror_once,
)


def configuration(tmp_path: Path) -> MirrorConfiguration:
    return MirrorConfiguration(
        host="203.0.113.10",
        port=22022,
        identity_file=tmp_path / "id_ed25519_runpod",
        known_hosts_file=tmp_path / "known_hosts",
        remote_dir="/workspace/results/portfolio-benchmark",
        local_dir=tmp_path / "results",
    )


def test_rsync_command_is_strict_and_never_deletes_local_results(
    tmp_path: Path,
) -> None:
    command = build_rsync_command(configuration(tmp_path))

    assert command[:4] == ("rsync", "--archive", "--partial", "--safe-links")
    assert "--delete" not in command
    assert "StrictHostKeyChecking=yes" in command
    assert "BatchMode=yes" in command
    assert command[-2] == "root@203.0.113.10:/workspace/results/portfolio-benchmark/"
    assert command[-1] == f"{tmp_path / 'results'}/"


@pytest.mark.parametrize(
    "remote_dir",
    ("results/run", "/workspace", "/workspace/results", "/etc/results"),
)
def test_configuration_rejects_unsafe_remote_directory(
    tmp_path: Path, remote_dir: str
) -> None:
    unsafe = MirrorConfiguration(
        host="203.0.113.10",
        port=22022,
        identity_file=tmp_path / "key",
        known_hosts_file=tmp_path / "known_hosts",
        remote_dir=remote_dir,
        local_dir=tmp_path / "results",
    )

    with pytest.raises(ValueError, match="/workspace/results"):
        build_rsync_command(unsafe)


def test_mirror_once_uses_argument_list_and_creates_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configuration(tmp_path)
    observed: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], *, check: bool) -> None:
        observed["command"] = command
        observed["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)

    mirror_once(config)

    assert config.local_dir.is_dir()
    assert isinstance(observed["command"], tuple)
    assert observed["check"] is True

