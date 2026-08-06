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
