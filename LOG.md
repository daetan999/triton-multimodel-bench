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

## 2026-10-01 - Session 5

### Goal

Repair the follow-up smoke-test failure from workflow run `36808489353`.

### Finding

Triton Server 2.59 prints its usage for `--help` but exits with status 1, so it cannot be used as a successful smoke-test command. The image again built successfully, and publication remained safely skipped.

### Fix

Use `command -v tritonserver` to verify the executable is installed without depending on Triton's CLI exit convention. The contract test now rejects both `--version` and `--help`.

## 2026-10-01 - Session 6

### Goal

Complete image publication and diagnose the final workflow-reporting failure in run `36811835631`.

### Result

- The Linux AMD64 image build passed.
- Triton presence, Python imports, missing-key rejection, and live SSH startup all passed.
- GHCR accepted every layer and published tag `0.1.0` with digest `sha256:b49adeeea3707504067b80f360af03952dc59286e5af3771baee7610321b566c`.
- The workflow was marked failed only after publication because `docker push` wrote its digest line to stderr while the parser captured stdout.

### Fix

Redirect Docker push stderr into stdout before `tee`, and add a contract-test assertion for that behavior. The already-tested and published image does not need to be rebuilt.

## 2026-10-01 - Session 7

### Goal

Verify public image access and prepare the exact paid deployment boundary.

### Verification

- An anonymous GHCR token fetched the `0.1.0` manifest successfully.
- The public manifest contains 42 layers and exactly matches digest `sha256:b49adeeea3707504067b80f360af03952dc59286e5af3771baee7610321b566c`.
- RunPod reported one Secure Cloud L4 at US$0.49/hour with low stock and CUDA 12.8, 13.0, and 13.2 availability.
- L4 stock appeared in `EU-RO-1`, `EUR-IS-1`, `US-GA-2`, and `US-MO-2`.
- The RunPod account still contains zero Pods and zero network volumes.

### Decision

Use no network volume. Allow all Secure Cloud locations, require an L4 with a CUDA 13.0 host floor, and use a 20 GB disposable container disk. The six-hour maximum compute cost is US$2.94 before the small temporary-disk charge.

## 2026-10-01 - Session 8

### Goal

Deploy and verify the first paid Secure Cloud L4 Pod.

### Result

- Created Pod `1fp0q1qajk3f0t` in `EU-RO-1` from the immutable project image digest.
- Allocated one NVIDIA L4 at US$0.49/hour with a 20 GB disposable container disk and no network volume.
- Exposed SSH only; Triton HTTP, gRPC, and metrics ports remain private.
- Recorded a manual termination target of `2026-10-01T10:00:00Z` (18:00 Singapore time).
- Automatic payments remain disabled by user confirmation; RunPod's MCP does not expose that billing setting for independent verification.

### Verification

- The container pulled successfully and SSH became reachable.
- `nvidia-smi` reported NVIDIA L4, 23,034 MiB VRAM, driver 595.91.07, and host CUDA 13.2.
- `/opt/tritonserver/bin/tritonserver` exists and is executable.
- ONNX, ONNX Runtime, Triton client, and XGBoost imports passed.
- `/workspace/model_repository`, `/workspace/results`, and `/workspace/logs` were created and are writable.
- RunPod reports 18 vCPUs and 71 GB RAM; the observed cgroup limits are 15.3 CPU cores and 70,999,998,464 bytes of memory.
- The first Triton launch rejected every FIL config because Triton 2.59 requires `output_class`; replaced the obsolete `is_classifier` parameter with `output_class: false`.
- The second launch reached artifact loading but rejected XGBoost 3.4.0 UBJSON and JSON with the embedded Treelite parser.
- Pinned XGBoost 3.0.2 and changed artifact creation from `XGBRegressor.save_model()` to direct Booster serialization.
- The final launch reported 20 of 20 models READY: 10 ONNX Runtime and 10 FIL.
- Real HTTP inference succeeded against `onnx_model_000` and `fil_model_000`; outputs `3.1630466` and `3.1630468` matched within `rtol=1e-5` and `atol=1e-5`.
- ONNX returned shape `[1, 1]` while FIL returned `[1]`; comparison therefore normalizes both outputs to one dimension.
- The live Pod was hot-patched with commit `63d0be3`; published image `0.1.0` does not contain these fixes.

### TDD evidence

- RED: `test_generate_repository_creates_one_complete_model_pair` failed because `output_class` was absent.
- GREEN: the same test passed after replacing `is_classifier`; checkpoint commits `abe9687` and `17a67e1`.
- RED: `test_generate_repository_serializes_the_booster_directly` failed because the generator called `XGBRegressor.save_model()`.
- RED: `test_pod_dependencies_match_the_triton_release` failed while `requirements-pod.txt` still pinned XGBoost 3.4.0.
- GREEN: all 11 tests passed under XGBoost 3.0.2 with 84% package coverage; checkpoint commits `d9f92b9`, `07bcdab`, and `63d0be3`.
- `pip check` passed and `pip-audit` found no known vulnerabilities in the pinned local and Pod requirements.

### Next step

Push the compatibility commits and publish image `0.1.1`. Verify its immutable digest before any formal 100-pair benchmark run.

## 2026-10-01 - Session 9

### Goal

Preserve the live L4 smoke-test evidence and end the paid Pod session safely.

### Result

- Captured Triton logs, the model repository index, package and GPU details, the
  generated model manifest, and a fresh paired ONNX/FIL inference result.
- Verified that all transferred files match the SHA-256 hashes generated on the
  Pod.
- Confirmed 20 READY models and matching paired inference outputs.
- Scanned the saved evidence for common credential markers; none were found.
- Terminated RunPod Pod `1fp0q1qajk3f0t`; a follow-up lookup returned HTTP 404.

### Next step

Push the complete change set and publish the corrected Pod image as `0.1.1`.

## 2026-10-01 - Session 10

### Goal

Publish the corrected, reproducible Pod image as `0.1.1`.

### Result

- Pushed the seven local compatibility and evidence commits to `main` at
  `15325f3` after all 11 tests passed.
- GitHub Actions run `36816708070` built the Linux AMD64 image and passed the
  Triton, Python, missing-key, and live SSH smoke checks.
- GHCR published `0.1.1` with digest
  `sha256:ad62d0e0006825b741238007ef364ed63a80a51bc4bd2b27499e8f5e1eaf5ad7`.
- An anonymous registry request returned HTTP 200 and a 42-layer manifest with
  the same digest, confirming that RunPod can pull it without credentials.
- The workflow's final reporting check failed because its `awk` expression read
  the word `digest:` instead of the following SHA. Added a failing regression
  assertion, replaced the positional parser with direct SHA-256 extraction, and
  returned the full suite to green.
- Corrected workflow run `36818347526` completed green and reported the same
  immutable digest.

### Next step

Deploy a fresh L4 Pod from the immutable `0.1.1` digest and run the formal
100-pair benchmark.
