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
| Pod ID | `1fp0q1qajk3f0t` |
| Cloud type | Secure Cloud |
| Data center | `EU-RO-1` |
| GPU | NVIDIA L4, 23,034 MiB usable VRAM |
| Host CPU | 18 vCPUs reported by RunPod; cgroup quota 15.3 cores; host cpuset exposes CPUs 0-127 |
| Host RAM | 71 GB reported by RunPod; cgroup limit 70,999,998,464 bytes (66.1 GiB) |
| GPU price and retrieval date | US$0.49/hour; low stock; retrieved 2026-10-01 |
| Live L4 data centers | `EU-RO-1`, `EUR-IS-1`, `EUR-IS-2`, `US-GA-2`, `US-MO-2` at retrieval time |
| Container disk | 20 GB disposable; confirmed on Pod |
| Network volume | None; account verified empty 2026-10-01 |
| Prepaid funding | Maximum US$10; user-confirmed 2026-10-01 |
| Automatic payments | Disabled; user-confirmed 2026-10-01 |
| Pod created at | `2026-10-01T04:14:30.678Z` (12:14:30 Singapore time) |
| Manual termination target | By `2026-10-01T10:00:00Z` (18:00 Singapore time) |
| Automatic termination | Unavailable; current RunPod backend does not enforce Pod deadlines |
| Pod status | Terminated 2026-10-01 after evidence transfer; follow-up lookup returned HTTP 404 |

## Triton server

| Field | Value |
|---|---|
| Container tag | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| Project image tag | `ghcr.io/daetan999/triton-multimodel-bench:0.1.1` |
| Project image digest | `sha256:ad62d0e0006825b741238007ef364ed63a80a51bc4bd2b27499e8f5e1eaf5ad7` |
| Live Pod hotfix | Baked into published image `0.1.1` |
| Triton server | 2.59.0 |
| CUDA in container | 12.9.1 |
| ONNX Runtime backend | 1.22.0 |
| FIL backend | Included |
| XGBoost | 3.0.2 in published image `0.1.1`; `0.1.0` contained incompatible 3.4.0 |
| NVIDIA driver | 595.91.07; host CUDA 13.2 |
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
