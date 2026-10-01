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

**Status:** Accepted
**Date:** 2026-08-06

Use one on-demand RunPod L4 Pod instead of Google Cloud. RunPod avoids the unavailable new-account L4 quota, uses prepaid credit, and preserves the project-defining Triton, CUDA, ONNX, and L4 stack.

Prefer Secure Cloud, read availability and price immediately before launch, add no more than US$10 prepaid credit, disable automatic payments, and create every Pod with a six-hour termination deadline.

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
