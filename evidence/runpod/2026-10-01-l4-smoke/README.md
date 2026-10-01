# RunPod L4 smoke-test evidence

This directory preserves the evidence copied from RunPod Pod `1fp0q1qajk3f0t`
before it was terminated on 2026-10-01.

- `environment.txt` records the GPU, driver, capture time, and package versions.
- `repository-index.json` records 20 READY models: 10 ONNX and 10 FIL.
- `inference-check.txt` records a real paired inference request and confirms that
  the flattened ONNX and FIL results match within the configured tolerance.
- `triton-10.log` records the original FIL configuration failure.
- `triton-10-fixed.log` records the XGBoost/Treelite compatibility failure.
- `triton-10-green.log` records the successful launch after both fixes.
- `triton-json-probe.log` records the intermediate artifact-format probe.
- `model-manifest.json` identifies the exact ten-pair generated repository.
- `SHA256SUMS` contains checksums generated on the Pod before transfer. The
  transferred files were verified byte-for-byte against these checksums.
- `ready.txt` is intentionally empty because Triton's successful readiness
  endpoint returns HTTP 200 with an empty body.

The evidence was scanned before commit for common credential markers. No API
keys, authorization headers, passwords, private keys, tokens, or secrets were
detected.
