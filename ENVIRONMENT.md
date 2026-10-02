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

## Diagnostic L4 Pod (terminated)

| Field | Value |
|---|---|
| Pod ID | `8ngraqlu5uc7v1` |
| Cloud type | Secure Cloud |
| Data center | `US-MO-2` |
| GPU | NVIDIA L4, 23,034 MiB usable VRAM |
| Host CPU | Session 12: 16 vCPUs reported by RunPod; cgroup quota 13.6 cores; host cpuset exposed CPUs 0-127 |
| Host RAM | 71 GB reported by RunPod; cgroup limit 70,999,998,464 bytes (66.1 GiB) |
| GPU price and retrieval date | US$0.49/hour; low stock; retrieved 2026-10-01 |
| Live L4 data centers | `EU-RO-1`, `EUR-IS-1`, `EUR-IS-2`, `US-GA-2`, `US-MO-2` at retrieval time |
| Container disk | 20 GB disposable; confirmed on Pod |
| Network volume | None; account verified empty 2026-10-01 |
| Prepaid funding | Maximum US$10; user-confirmed 2026-10-01 |
| Automatic payments | Disabled; user-confirmed 2026-10-01 |
| Pod created at | `2026-10-01T06:13:12.278Z` (14:13:12 Singapore time) |
| Manual termination target | By `2026-10-01T12:13:12Z` (20:13:12 Singapore time) |
| Automatic termination | Unavailable; current RunPod backend does not enforce Pod deadlines |
| Pod status | Terminated after verified diagnostic evidence transfer; follow-up lookup returned HTTP 404 |

## L40S retry (terminated; measured matrix not retained)

| Field | Value |
|---|---|
| Pod ID | `jfzwlzxfxxfa8o` |
| Cloud type | Secure Cloud |
| Data center | `US-TX-4` |
| GPU | NVIDIA L40S, 46,068 MiB usable VRAM |
| GPU price | US$1.09/hour |
| Host CPU | 16 vCPUs reported by RunPod; cgroup quota 13.6 cores; cpuset exposed CPUs 0-127 |
| CPU split | Triton CPUs 0-7; load generator CPUs 8-12 |
| NVIDIA driver | 595.91.07 |
| Container disk | 20 GB disposable |
| Network volume | None |
| Pod created at | `2026-10-01T07:13:36.868Z` |
| Verified preflight | 100 models loaded; 257 measured requests; 51.4 achieved QPS; zero failures |
| Full-matrix observation | At least 32 of 81 summaries observed valid before monitoring was interrupted |
| Retained result status | Not retained; the disposable disk was deleted with the Pod |
| Audited Pod cost | US$9.46550393011421 |
| Current account resources | Zero Pods and zero network volumes, verified 2026-10-02 |

## Final L4 portfolio benchmark (terminated)

| Field | Value |
|---|---|
| Pod ID | `2xaiz7x6p3dybr` |
| Cloud / data center | Secure Cloud / `EUR-IS-1` |
| GPU | NVIDIA L4, 23,034 MiB usable VRAM |
| NVIDIA driver / RunPod CUDA | 580.159.04 / 13.0 |
| Host CPU | AMD EPYC 7713; 128 logical CPUs visible; cgroup quota 15.3 cores |
| CPU split | Triton CPUs 0–7; load generator CPUs 8–12 |
| Live price | US$0.49/hour, retrieved 2026-10-02 |
| Container disk / network volume | 20 GB disposable / none |
| Pod created at | `2026-10-02T01:46:42.191Z` |
| Container-ready uptime at final read | 989 seconds; image download time preceded container readiness |
| Approved maximum | Two hours; US$0.98 compute plus the temporary disk charge |
| Approximate wall-clock compute cost | About US$0.30 at the quoted hourly rate |
| Preflight | 100 models; 257 measured requests; 51.4 achieved QPS; zero failures |
| Formal matrix | 18/18 valid; 107,988 measured requests; zero failures |
| Evidence | 83 formal-matrix files verified against Pod-generated SHA-256 checksums |
| Final resource state | Pod terminated; zero Pods and zero network volumes verified 2026-10-02 |

## Triton server

| Field | Value |
|---|---|
| Container tag | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| Project image tag | `ghcr.io/daetan999/triton-multimodel-bench:0.1.4` |
| Project image digest | `sha256:50087af8c9182b01fd1d45c6c4c7d77fffe0c321f2e6829b49d48a0add3cccbb` |
| Image source commit | `0c40a1fda6bc6bd2989f07d8daada8906fdd3efc` |
| Release history | FIL compatibility in `0.1.1`; harness in `0.1.2`; bounded ONNX threads in `0.1.3`; result mirroring support in `0.1.4` |
| Triton server | 2.59.0 |
| CUDA in container | 12.9.1 |
| ONNX Runtime backend | 1.22.0 |
| FIL backend | Included |
| XGBoost | 3.0.2 from image `0.1.1` onward; `0.1.0` contained incompatible 3.4.0 |
| Final benchmark driver | 580.159.04; RunPod CUDA 13.0 |
| ONNX opset | 15 |

## Benchmark policy

| Field | Value |
|---|---|
| Model counts | 1, 10, 100 |
| Batching conditions | Off, 10 ms |
| Offered load | 200 requests/second |
| Repetitions | 3 per condition |
| Warm-up | Recorded separately; excluded from measured window |
| Raw data | One row per request |
| Load generator | Same Pod, disjoint CPU affinity from Triton |
| Network path | Loopback (`127.0.0.1`); not a network benchmark |
| Final matrix size | 18 runs: 3 model counts × 2 batching policies × 3 repetitions |

## Generated model repository

| Field | Value |
|---|---|
| Model pairs | 100 |
| Training seed | 20260805 |
| Features per request | 32 FP32 values |
| ONNX representation | XGBoost regressor converted at opset 15 |
| FIL representation | Same XGBoost regressor serialized as UBJ |
| Local validation | Passed 2026-10-01 |
