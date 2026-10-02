# Decision record

## D001 - Use NVIDIA L4 rather than T4

**Status:** Accepted
**Date:** 2026-08-05

Use one NVIDIA L4 with 24 GB GPU memory on RunPod. Record the host CPU, RAM, cloud type, and data center from the actual Pod because RunPod host configurations vary.

The L4 is a current inference-focused data-centre GPU. The project is a modern synthetic analogue of the earlier architecture, not a hardware-identical reproduction.

## D002 - Use a separate load generator

**Status:** Superseded by D007
**Date:** 2026-08-05

Use an `n2-standard-4` VM in the same zone to generate traffic. Reusing it for the CPU baseline limits cost while preventing the load generator from consuming the GPU server's CPU resources.

## D003 - Pin Triton 25.06

**Status:** Accepted
**Date:** 2026-08-05

Use `nvcr.io/nvidia/tritonserver:25.06-py3`. It contains Triton 2.59.0, CUDA 12.9.1, ONNX Runtime 1.22.0, and FIL. Pin the derived project image by digest and verify its driver compatibility on the real RunPod host before benchmarking.

## D004 - Never expose Triton publicly

**Status:** Accepted
**Date:** 2026-08-05

Triton ports 8000, 8001, and 8002 remain private. The load generator connects through `127.0.0.1` inside the Pod. The Pod exposes only SSH; no RunPod HTTP proxy or public TCP mapping is created for Triton.

## D005 - Pin ONNX opset 15

**Status:** Accepted
**Date:** 2026-08-05

Use ONNX opset 15 for both scikit-learn and LightGBM exports. The installed LightGBM converter rejects higher target opsets; one shared opset prevents the model zoo from mixing formats.

## D006 - Switch the execution provider from Google Cloud to RunPod

**Status:** Accepted; cost-guard clause superseded by D013
**Date:** 2026-08-06

Use one on-demand RunPod L4 Pod instead of Google Cloud. RunPod avoids the unavailable new-account L4 quota, uses prepaid credit, and preserves the project-defining Triton, CUDA, ONNX, and L4 stack.

Prefer Secure Cloud, read availability and price immediately before launch, add no more than US$10 prepaid credit, and disable automatic payments.

## D007 - Use a CPU-isolated single-node load generator

**Status:** Accepted
**Date:** 2026-08-06

Run the benchmark client on the L4 Pod but pin it to CPU cores that Triton cannot use. Record both CPU affinity masks and reject runs where the client CPU set is saturated.

This is a controlled single-node microbenchmark, not a client-to-server network benchmark. It reduces cost and removes public-network variance, but it can still understate contention compared with a truly separate load generator. The limitation must appear in the README and final report.

## D008 - Generate equivalent XGBoost model pairs

**Status:** Accepted
**Date:** 2026-10-01

Train one deterministic XGBoost regressor per model index, export that trained model to ONNX for the ONNX Runtime backend, and serialize the same model as XGBoost UBJ for the FIL backend.

The local validator checks artifact hashes, Triton configurations, ONNX validity, and paired predictions. This makes backend comparisons about execution rather than differences between independently trained models.

## D009 - Publish a minimal project-specific Pod image through GitHub Actions

**Status:** Accepted
**Date:** 2026-10-01

Derive a Linux AMD64 image from `nvcr.io/nvidia/tritonserver:25.06-py3`, add only key-authenticated SSH and pinned project runtime dependencies, and publish versioned images to GitHub Container Registry. Never publish `latest`.

The Mac does not have Docker installed. A manually triggered GitHub Actions workflow therefore builds and smoke-tests the image before publishing it. RunPod must use the resulting immutable digest, not the mutable semantic tag. Triton ports remain unexposed.

## D010 - Use disposable Pod storage instead of a network volume

**Status:** Superseded by D013
**Date:** 2026-10-01

Use a 20 GB container disk and no network volume. The model repository is reproducible, the benchmark runs on one Pod, and results will be copied to the Mac before termination. Removing persistent storage avoids an independently billed resource and lets RunPod place the low-stock L4 in any available Secure Cloud data center.

At current published rates, 20 GB of standard network storage would cost about US$1.40 per month until deleted. Disposable Pod storage is sufficient for this workload and disappears with the Pod.

## D011 - Use a recorded manual Pod cutoff

**Status:** Superseded by D013
**Date:** 2026-10-01

Record a manual cutoff no later than six hours after each Pod creation and terminate the Pod explicitly at the end of the session. Keep automatic payments disabled so the prepaid balance is the account-level funding boundary.

RunPod removed its `--terminate-after` and `--stop-after` controls after confirming that the backend accepted but did not enforce them. The current Pod API exposes no reliable automatic deadline, so the runbook must not present one as a safety mechanism.

## D012 - Use a bounded global ONNX Runtime thread pool

**Status:** Accepted
**Date:** 2026-10-01

Launch Triton's ONNX Runtime backend with one shared global thread pool, eight intra-op threads, one inter-op thread, and four model-loading threads. Apply the same launch settings to every ONNX model-count condition.

The first formal attempt used ONNX Runtime's defaults, which created a per-session pool based on 128 visible host CPUs. Triton aborted while loading model 69 of 100 with `std::system_error: Resource temporarily unavailable`. NVIDIA's ONNX Runtime backend documents the global pool specifically for multi-session deployments. Bounding both execution and loading threads allows the 100-model configuration to fit the Pod while keeping Triton pinned to CPUs 0–7.

## D013 - Mirror results continuously and schedule an external termination check

**Status:** Accepted
**Date:** 2026-10-02

Keep the reproducible models on disposable Pod storage, but continuously mirror every benchmark artifact to the Mac while a run is active. A canary copy must succeed before measurement begins, and the local mirror must never delete previously copied files.

Before creating a paid Pod, record the maximum quoted cost and arrange an independently scheduled termination check. A remembered or verbally stated timestamp is not a hard cutoff. These controls replace the disposable-only and manual-cutoff assumptions in D010 and D011.

## D014 - Reduce the final experiment to a portfolio-sized matrix

**Status:** Accepted
**Date:** 2026-10-02

Run 18 ONNX measurements: 1, 10, and 100 loaded models; batching off and a 10 ms batching window; 200 requests per second; and three repetitions. Use five seconds of warm-up and 30 seconds of measured traffic.

The earlier 81-run plan was broader than needed for the GitHub and resume goal. The smaller matrix retains repeated measurements of the two project questions—model-count overhead and the latency/throughput effect of batching—while reducing paid GPU time. It must not be described as an exhaustive Triton performance study.
