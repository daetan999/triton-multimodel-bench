# Build log

Add one entry at the end of every working session. Do not rewrite old entries.

## 2026-08-05 - Session 0

### Goal

Prepare the repository and local tooling.

### Completed

- Created the public GitHub repository.
- Cloned the repository locally.
- Installed Google Cloud CLI 579.0.0.
- Installed OpenMP runtime `libomp` 22.1.8 for LightGBM on Apple Silicon.
- Added the initial repository structure and runbook.
- Selected Google Cloud `g2-standard-4` with one NVIDIA L4 as the benchmark server.
- Selected Triton container `nvcr.io/nvidia/tritonserver:25.06-py3` as the compatibility baseline.

### Verification

- Created `.venv` with Python 3.12 and installed the pinned requirements.
- Initial LightGBM import exposed the missing OpenMP runtime; installed `libomp` and retained the failure in this log.
- The first export check showed that the LightGBM converter rejects ONNX opset 20; pinned all exports to opset 15.
- Verified scikit-learn and LightGBM opset-15 exports by executing them with ONNX Runtime.
- Added repeatable environment tests for both conversion paths.
- Final verification: two tests passed and `pip check` reported no broken requirements.
- Pending Google Cloud authentication, billing, and L4 quota approval.

### Next step

Follow Phase 1 in `RUNBOOK.md` and stop after the readiness check.

## 2026-08-06 - Session 1

### Goal

Replace the blocked Google Cloud execution plan with a safe RunPod workflow and authenticate Codex.

### Completed

- Installed and authenticated the official `runpod@runpod` Codex plugin.
- Confirmed that `codex mcp list` reports the RunPod MCP as enabled with OAuth.
- Replaced the GCP quota, VM, billing, and teardown instructions with a RunPod L4 Pod workflow.
- Set a US$10 prepaid funding limit, disabled automatic payments as a requirement, and required a six-hour Pod termination deadline.
- Changed the benchmark to an explicitly documented single-node design with disjoint CPU affinities for Triton and the load generator.
- Kept Triton ports private and required Pod termination rather than stopping.

### Verification

- A mistaken Flash CLI login attempt failed and saved no RunPod API key; Flash is not part of the project setup path.
- The official RunPod MCP OAuth flow then completed successfully.
- No Pod, network volume, endpoint, or other paid RunPod resource creation command was issued.
- Documentation diffs passed `git diff --check`.

### Next step

Complete Phase 1.2 through 1.4 in `RUNBOOK.md`, then open a fresh Codex task to verify that the RunPod account has no existing Pods or network volumes. Do not create paid resources before ten model pairs pass locally.

## 2026-10-01 - Session 2

### Goal

Build and verify the deterministic paired model repository before renting GPU infrastructure.

### Completed

- Added `triton_benchmark.model_zoo` as the public generation and validation module.
- Added command-line wrappers at `scripts/gen_models.py` and `scripts/validate_models.py`.
- Trained one XGBoost regressor per index and exported the same trained model to ONNX and XGBoost UBJ.
- Generated Triton ONNX Runtime and FIL configurations with fixed tensor names, shapes, and GPU instance groups.
- Added a manifest containing seeds and SHA-256 hashes for every artifact.
- Ignored the complete generated model repository while retaining its `.gitkeep` placeholder.

### Verification

- Added behavior tests for repository layout, execution validation, deterministic hashes, and tamper detection.
- The determinism test exposed random ONNX graph UUIDs; normalized each graph name to its model name.
- The ten-pair gate passed with 10 ONNX models and 10 FIL models.
- The full gate passed with 100 ONNX models and 100 FIL models.
- Coverage passed at 84% for the `triton_benchmark` package with all six tests passing.
- A live read confirmed that the RunPod account contains zero Pods and zero network volumes.
- RunPod confirmed one registered ED25519 public key labeled `runpod-triton-benchmark`; its fingerprint matches the dedicated local key.
- User confirmed a maximum US$10 prepaid balance and that automatic payments are disabled; RunPod's MCP does not expose either account setting for independent verification.
- RunPod billing reported US$0 resource spend for the seven-day window ending 2026-10-02, with zero Pods and zero network volumes.

### Next step

Review, commit, and push the local changes, then run the manual GitHub image workflow in Phase 3.2. Do not create paid infrastructure until the image passes its smoke test and is public by immutable digest.

## 2026-10-01 - Session 3

### Goal

Prepare a reproducible and securely accessible Triton Pod image without starting paid infrastructure.

### Completed

- Added a Linux AMD64 Dockerfile derived from `nvcr.io/nvidia/tritonserver:25.06-py3`.
- Added key-only SSH startup that refuses to run without a valid injected public key.
- Added a minimal, pinned Pod dependency set matching Triton client 2.59.0.
- Added a Docker context denylist for credentials, local environments, generated models, and benchmark results.
- Added a manually triggered GitHub Actions workflow that builds, smoke-tests, and publishes only a semantic version to GitHub Container Registry.
- Pinned every third-party GitHub Action to a full commit SHA and made the workflow print the immutable image digest.

### Verification

- Docker is not installed on the local Mac, so no local image build was attempted.
- All four container-contract tests passed.
- The Pod entrypoint passed Bash syntax validation.
- No RunPod Pod, volume, or other paid resource was created.

### Next step

Review, commit, and push the repository changes. Then run **Publish Pod image** with tag `0.1.0`, make the resulting GHCR package public, and record its digest before creating a Pod.

## 2026-10-01 - Session 4

### Goal

Diagnose and repair the failed Pod image publication workflow.

### Finding

The image built successfully. The smoke test failed because Triton Server 2.59 does not implement a `--version` option; it printed usage and exited with status 1 before the Python and SSH checks ran.

### Fix

Replaced the unsupported option with `tritonserver --help`, which verifies that the executable is present and runnable. Added a workflow-contract regression assertion that rejects `--version`.
