# Triton Multi-Model Benchmark

Reproducible benchmark of multi-model NVIDIA Triton inference on an NVIDIA L4 GPU, measuring dynamic batching, backend performance, latency, GPU utilisation, and cost.

This repository is under active development. Results will not be published until every reported number can be traced to raw data, a command, a configuration, and a recorded environment.

Start with [RUNBOOK.md](RUNBOOK.md).

## Planned experiments

1. Dynamic batching across traffic distributed over 1, 10, and 100 models.
2. ONNX Runtime versus FIL for equivalent tree-model workloads.
3. GPU consolidation cost versus a four-vCPU CPU baseline.
4. Optional cold-load and GPU-memory measurements.

## Repository layout

```text
configs/            Experiment definitions and Triton configurations
model_repository/   Generated Triton model repository
results/raw/        Per-request measurements
results/gpu/        GPU telemetry
results/charts/     Final charts
scripts/            Model generation, load generation, and analysis
tests/              Automated checks
```
