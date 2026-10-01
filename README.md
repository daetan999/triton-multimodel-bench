# Triton Multi-Model Benchmark

Reproducible benchmark of multi-model NVIDIA Triton inference on an NVIDIA L4 GPU, measuring dynamic batching, backend performance, latency, GPU utilisation, and cost.

This repository is under active development. Results will not be published until every reported number can be traced to raw data, a command, a configuration, and a recorded environment.

The benchmark will run on one on-demand RunPod L4 Pod. Triton and the load generator will use recorded, disjoint CPU affinities on the same host, so this is a single-node inference microbenchmark rather than a client-to-server network benchmark.

Start with [RUNBOOK.md](RUNBOOK.md).

## Generate and validate the model zoo

```bash
source .venv/bin/activate
python scripts/gen_models.py --count 100 --seed 20260805
python scripts/validate_models.py --count 100
```

Each index produces two representations of the same trained XGBoost regressor: an ONNX model for Triton's ONNX Runtime backend and an XGBoost UBJ model for Triton's FIL backend. Generated artifacts stay out of Git because the manifest, model files, and Triton configurations are reproducible from the pinned code and seed.

## Build the Pod image

The manually triggered **Publish Pod image** GitHub Actions workflow builds the pinned Linux AMD64 image, smoke-tests Triton and SSH startup, and publishes a semantic version to GitHub Container Registry. Follow Phase 3.2 of `RUNBOOK.md`; deploy only the digest-pinned reference printed in the workflow summary.

## Planned experiments

1. Dynamic batching across traffic distributed over 1, 10, and 100 models.
2. ONNX Runtime versus FIL for equivalent tree-model workloads.
3. GPU consolidation cost versus a four-vCPU CPU baseline.
4. Optional cold-load and GPU-memory measurements.

## Benchmark runner

The runner sends asynchronous gRPC requests on a deterministic Poisson arrival schedule and chooses uniformly across the active models. Warm-up requests remain in the raw CSV but are excluded from measured metrics. Every formal run records per-request latency, target and achieved QPS, errors, one-second GPU telemetry, Triton process CPU/RSS telemetry, the immutable image digest, Git commit, and CPU affinities.

The first formal matrix is 81 ONNX Runtime runs: 1/10/100 models, batching off/2 ms/10 ms, 50/200/500 target QPS, and three repetitions. `scripts/run_matrix.py` writes an immutable matrix manifest and only resumes complete, validated runs.

## Repository layout

```text
configs/            Experiment definitions and Triton configurations
model_repository/   Generated Triton model repository
results/raw/        Per-request measurements
results/gpu/        GPU telemetry
results/cpu/        Triton process CPU and memory telemetry
results/summaries/  Validated run metadata and aggregate metrics
results/server/     Triton logs grouped by server configuration
results/charts/     Final charts
scripts/            Model generation, load generation, and analysis
tests/              Automated checks
```
