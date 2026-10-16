# V2 Plan — Saturation and Batching Crossover Study

V1 answered one narrow question: at roughly 200 QPS, does a 10 ms Triton batching window help when one NVIDIA L4 serves up to 100 small ONNX models?

For that workload and operating point, it did not. The queue increased p95 latency by about 9.4× while delivered throughput stayed broadly unchanged.

V2 is designed to answer the question V1 deliberately left open:

> **At what load does batching begin to earn back its queueing delay, and where does the L4 actually saturate?**

## Scope

V2 will extend the existing benchmark harness rather than create a separate project. V1 results and evidence remain frozen; V2 will be stored as a new experiment series with its own manifests, raw request records, telemetry, summaries, validation output, checksums, and environment capture.

### Primary experiment

The main V2 study will use the 100-model condition as the primary consolidation case and sweep load upward until the system reaches a clear saturation or service-level boundary.

Planned request-rate ladder:

- 50 QPS
- 100 QPS
- 200 QPS
- 400 QPS
- 800 QPS
- 1,200 QPS
- higher rates if the system remains below saturation

Planned batching windows:

- off / 0 ms
- 1 ms
- 2 ms
- 5 ms
- 10 ms

The exact upper request rate may change during execution. The goal is not to force a predetermined maximum; it is to continue until the benchmark identifies the useful operating boundary.

### Measurements

V2 will continue to capture per-request results and will compare each policy using:

- achieved QPS;
- p50, p95, and p99 latency;
- failed or timed-out requests;
- GPU memory allocation;
- finer-grained GPU utilisation telemetry than V1;
- CPU and Triton process resource usage; and
- queueing/batching behaviour where the Triton metrics expose it reliably.

Where the environment and price basis are sufficiently controlled, V2 will also derive a simple serving-efficiency view such as delivered requests per GPU-hour / cost per million delivered requests. Any commercial metric will be kept separate from the measured systems metrics and will state its pricing assumptions.

## Decision points

V2 is intended to identify four things rather than simply produce a larger table:

1. **Saturation point** — where offered load stops translating cleanly into delivered throughput, latency breaks sharply, or failures begin to rise.
2. **Batching crossover** — the load at which a batching policy produces enough throughput benefit to justify its additional queueing latency.
3. **Useful operating region** — the range where latency, throughput, and reliability remain acceptable together.
4. **Consolidation behaviour** — whether the V1 observation of stable low-latency serving with 100 loaded models continues as the GPU is pushed harder.

## Method changes from V1

V2 will keep the reproducibility controls that mattered in V1:

- deterministic model generation;
- asynchronous gRPC load generation;
- warm-up excluded from measured results;
- CPU isolation between Triton and the load generator;
- immutable container provenance;
- repeated runs per condition;
- raw per-request output;
- environment capture;
- validation reports and SHA-256 checksums.

The main methodological changes are:

- sweep request rate instead of holding it at 200 QPS;
- test several queue-delay values instead of only 0 and 10 ms;
- increase GPU telemetry resolution so short accelerator bursts are less likely to disappear between samples; and
- define explicit saturation / crossover criteria before interpreting the result.

## Planned output

The final V2 write-up should make one chart especially clear:

**offered load → delivered QPS and tail latency for each batching policy**

The objective is to make the crossover visible rather than argue from a single operating point.

V1 remains the baseline. V2 should either show where batching starts helping, or show that this particular small-model workload needs a different optimisation before batching becomes worthwhile.
