# L4 benchmark diagnostic run

Pod `8ngraqlu5uc7v1` ran image `0.1.2` in RunPod Secure Cloud `US-MO-2` at US$0.49/hour.

The five-second preflight passed, followed by 18 complete and valid 10-model runs. Triton then aborted while loading ONNX model 69 of the first 100-model group:

```text
terminate called after throwing an instance of 'std::system_error'
  what():  Resource temporarily unavailable
```

The Pod exposed 128 host CPUs but had a 13.6-core cgroup quota. ONNX Runtime's default per-session pools therefore created too many threads across 100 model sessions. The follow-up release uses one bounded global ONNX Runtime pool and four model-loading threads.

The complete local-only result set is retained beside this file. Every transferred file passed the Pod-generated `SHA256SUMS`. The compressed archive hash is:

```text
4d524f6593759afbfe4807468b3ffba4420b8640a05c1c774e63222ff75894ee  diagnostic-results.tar.gz
```

The Pod was terminated after transfer. RunPod then reported HTTP 404 for the Pod, zero Pods, and zero network volumes.
