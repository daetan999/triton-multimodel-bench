# Environment record

Do not fill fields from memory. Copy values from the commands in `RUNBOOK.md`.

## Local machine

| Field | Value |
|---|---|
| Operating system | macOS 14.5 |
| Architecture | arm64 |
| Python | 3.12.12 |
| Git | 2.39.5 |
| Codex RunPod MCP | OAuth authenticated; verify again in a fresh task |
| RunPod SSH key | `runpod-triton-benchmark`, fingerprint `SHA256:NPwsxwH8UOyjYMu6rgf9f3HbZV0yT/EDmtBrXQklddQ` |

## RunPod

| Field | Value |
|---|---|
| Pod ID | Pending |
| Cloud type | Secure preferred; pending actual value |
| Data center | Pending live availability check |
| GPU | NVIDIA L4, 24 GB |
| Host CPU | Pending actual Pod value |
| Host RAM | Pending actual Pod value |
| GPU price and retrieval date | Pending |
| Network volume ID | Pending |
| Network volume | 20 GB standard, mounted at `/workspace` |
| Prepaid funding | Maximum US$10; user-confirmed 2026-10-01 |
| Automatic payments | Disabled; user-confirmed 2026-10-01 |
| Pod created at | Pending |
| Automatic termination at | Pending; maximum six hours after creation |

## Triton server

| Field | Value |
|---|---|
| Container tag | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| Project image tag | `ghcr.io/daetan999/triton-multimodel-bench:0.1.0` |
| Project image digest | `sha256:b49adeeea3707504067b80f360af03952dc59286e5af3771baee7610321b566c` |
| Triton server | 2.59.0 |
| CUDA in container | 12.9.1 |
| ONNX Runtime backend | 1.22.0 |
| FIL backend | Included |
| NVIDIA driver | Pending `nvidia-smi` output |
| ONNX opset | 15 |

## Benchmark policy

| Field | Value |
|---|---|
| Model counts | 1, 10, 100 |
| Batching conditions | Off, 2 ms, 10 ms |
| Repetitions | At least 3 per condition |
| Warm-up | Recorded separately; excluded from measured window |
| Raw data | One row per request |
| Load generator | Same Pod, disjoint CPU affinity from Triton |
| Network path | Loopback (`127.0.0.1`); not a network benchmark |

## Generated model repository

| Field | Value |
|---|---|
| Model pairs | 100 |
| Training seed | 20260805 |
| Features per request | 32 FP32 values |
| ONNX representation | XGBoost regressor converted at opset 15 |
| FIL representation | Same XGBoost regressor serialized as UBJ |
| Local validation | Passed 2026-10-01 |
