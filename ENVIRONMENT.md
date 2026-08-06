# Environment record

Do not fill fields from memory. Copy values from the commands in `RUNBOOK.md`.

## Local machine

| Field | Value |
|---|---|
| Operating system | macOS 14.5 |
| Architecture | arm64 |
| Python | 3.12.12 |
| Google Cloud CLI | 579.0.0 |
| Git | 2.39.5 |

## Google Cloud

| Field | Value |
|---|---|
| Project ID | Pending |
| Region | `asia-southeast1` |
| Zone | `asia-southeast1-a` |
| GPU VM | `g2-standard-4` |
| GPU | NVIDIA L4, 24 GB |
| CPU/load-generator VM | `n2-standard-4` |
| Deep Learning VM image | Pending exact image name |
| GPU price and date | Pending |
| CPU price and date | Pending |

## Triton server

| Field | Value |
|---|---|
| Container tag | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| Container digest | Pending |
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
| Load generator | Separate VM in the same zone |
