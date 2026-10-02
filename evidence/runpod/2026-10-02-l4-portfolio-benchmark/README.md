# L4 portfolio benchmark evidence

This directory preserves the evidence copied from RunPod Pod `2xaiz7x6p3dybr` before it was terminated on 2026-10-02.

- `environment.txt` records the hardware, immutable image, benchmark commit, CPU quota, and runtime details.
- `results/preflight` contains the 100-model preflight that passed before the formal matrix began.
- `results/portfolio-benchmark` contains the 18-run formal matrix: raw request records, GPU and CPU telemetry, Triton logs, summaries, validation counts, and checksums.
- `SHA256SUMS` covers 83 files in the formal matrix archive. Every checksum was verified locally after the final mirror completed.

The formal archive reports 18 summaries, 18 raw request CSVs, 18 GPU traces, 18 CPU traces, six server logs, zero invalid runs, and zero failed measured requests.
