from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pod_image_uses_the_pinned_triton_base_and_secure_ssh() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ssh_config = (ROOT / "docker" / "sshd_config").read_text(encoding="utf-8")
    entrypoint = (ROOT / "docker" / "start-pod.sh").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM nvcr.io/nvidia/tritonserver:25.06-py3\n")
    assert 'ENTRYPOINT ["/usr/local/bin/start-pod"]' in dockerfile
    assert "PasswordAuthentication no" in ssh_config
    assert "KbdInteractiveAuthentication no" in ssh_config
    assert "PermitRootLogin prohibit-password" in ssh_config
    assert "PUBLIC_KEY" in entrypoint
    assert "authorized_keys" in entrypoint
    assert "exec /usr/sbin/sshd -D -e" in entrypoint


def test_pod_dependencies_match_the_triton_release() -> None:
    requirements = (ROOT / "requirements-pod.txt").read_text(encoding="utf-8")

    assert "tritonclient[grpc,http]==2.59.0" in requirements
    assert "xgboost==3.4.0" in requirements
    assert "onnx==1.22.0" in requirements


def test_docker_context_excludes_credentials_and_generated_data() -> None:
    exclusions = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert ".git" in exclusions
    assert ".env" in exclusions
    assert "*.key" in exclusions
    assert "*.pem" in exclusions
    assert "model_repository" in exclusions
    assert "results" in exclusions


def test_image_publish_workflow_is_manual_amd64_and_versioned() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish-pod-image.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "packages: write" in workflow
    assert "platforms: linux/amd64" in workflow
    assert "tags: ${{ steps.image.outputs.reference }}" in workflow
    assert ":latest" not in workflow
    assert "command -v tritonserver" in workflow
    assert "tritonserver --help" not in workflow
    assert "tritonserver --version" not in workflow
    assert "uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in workflow
