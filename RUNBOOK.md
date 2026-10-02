# Triton Multi-Model Benchmark Runbook

This is the operating guide for the whole project. Work from top to bottom. Do not skip a verification box.

## Where we are now

- [x] Repository created and cloned.
- [x] Python 3.12 environment created.
- [x] Pinned packages installed and checked.
- [x] scikit-learn and LightGBM ONNX export tests passing.
- [x] RunPod's official Codex plugin installed.
- [x] RunPod MCP OAuth completed.
- [x] Cloud plan changed from Google Cloud to RunPod.
- [x] RunPod account verified empty: zero Pods and zero network volumes.
- [x] RunPod account funded with a maximum of US$10 prepaid credit (user-confirmed).
- [x] Automatic payments disabled (user-confirmed).
- [x] Ten reproducible model pairs generated and validated locally.
- [x] Full 100-pair model repository generated and validated locally.
- [x] Project-specific Triton Pod image and publish workflow prepared locally.
- [x] Image `0.1.0` built, smoke-tested, and pushed to GHCR by digest.
- [x] GHCR image verified publicly pullable by immutable digest.
- [x] Live Secure Cloud L4 stock and price checked on 2026-10-01.
- [x] Paid L4 Pod deployed and its real CUDA, GPU, CPU, RAM, disk, and SSH environment verified.
- [x] Ten-pair Triton smoke gate passed after the live FIL compatibility fixes.
- [x] Smoke-test evidence copied locally and the first paid Pod terminated.
- [x] Corrected image `0.1.1` built, smoke-tested, published, and anonymously verified by digest.
- [x] Asynchronous Poisson gRPC load generator implemented and tested.
- [x] Crash-safe 81-run matrix runner implemented with per-request, GPU, CPU, and summary artifacts.
- [x] Image `0.1.3` published with bounded ONNX Runtime thread pools.
- [x] A 100-model L40S preflight completed successfully with zero failed requests.
- [x] All RunPod Pods and network volumes terminated; billing is stopped.
- [x] Continuous off-Pod result mirroring implemented and tested locally.
- [x] Image `0.1.4` published and verified by immutable digest.
- [x] Compact L4 preflight passed with 100 loaded models and zero failures.
- [x] Final 18-run portfolio matrix completed with 18 valid summaries and zero failed measured requests.
- [x] All 83 mirrored result files verified against Pod-generated SHA-256 checksums.
- [x] Final L4 Pod terminated; account re-verified at zero Pods and zero network volumes.
- [x] Results, chart, beginner-friendly explanation, evidence links, and resume bullets published in the README.

The project is complete at portfolio scope. The retained L4 matrix contains 18 valid runs, 107,988 measured requests, and zero failures. Its raw records, telemetry, logs, summaries, environment record, and checksums are committed under `evidence/runpod/2026-10-02-l4-portfolio-benchmark`. No paid RunPod resources remain.

## How to use this runbook

- Run one command block at a time.
- Unless a section says otherwise, run local commands from the repository root.
- Compare your output with **Success looks like** before continuing.
- If a command fails, copy the complete command and complete error into the working chat.
- Never substitute invented benchmark values for missing measurements.
- Never create a paid Pod without an independently scheduled termination check, a tested live result mirror, and explicit approval of the maximum quoted cost.
- Terminate Pods when a session ends; stopping a Pod can leave storage charges running.

## Fixed project choices

| Item | Choice |
|---|---|
| Cloud | RunPod, on-demand; Secure Cloud preferred |
| Data center | Any live Secure Cloud L4 location; do not pin without a storage or residency need |
| GPU server | One inference-capable NVIDIA GPU; prefer L4 for cost or L40S for availability, and record the exact device |
| Load generator | Same Pod, pinned to reserved CPU cores |
| CPU baseline | Same Pod CPU, with GPU execution disabled |
| Working storage | 20 GB disposable container disk continuously mirrored to the Mac |
| Triton container | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| ONNX opset | 15 |
| Maximum Pod run | Two hours for the compact benchmark; independently scheduled termination check required |
| Additional project funding | None unless the user explicitly approves a new quoted amount |

---

# Phase 0 - Local setup

Codex has already created the repository scaffold and verified the local Python environment.

## 0.1 Open the repository

Run on your Mac:

```bash
cd "/Users/dae/Documents/Codex/2026-08-05/i-w/outputs/triton-multimodel-bench"
pwd
```

**Success looks like:** the final line ends with `/triton-multimodel-bench`.

## 0.2 Create the Python environment

```bash
brew install libomp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This may take several minutes.

Verify:

```bash
python - <<'PY'
import lightgbm
import numpy
import onnx
import onnxmltools
import onnxruntime
import pandas
import sklearn
import skl2onnx
import xgboost

print("Local Python environment: READY")
PY
python -m pip check
coverage run -m pytest -q
coverage report
```

**Success looks like:** `Local Python environment: READY`, `No broken requirements found`, all tests pass, and total coverage is at least 80%.

## 0.3 Confirm repository state

```bash
git status --short
```

**Success looks like:** no unexpected files appear. Do not commit or push unreviewed changes.

---

# Phase 1 - RunPod account and safety setup

No GPU or storage resource is created in this phase.

## 1.1 Authenticate the official Codex plugin

This step is complete. Verify it at any time:

```bash
codex mcp list | sed -n '/^Name    Url/,$p'
```

**Success looks like:** `runpod`, `enabled`, and `OAuth` appear on one line.

If it says `Not logged in`, run:

```bash
codex mcp login runpod
```

Approve the page for the `runpod` MCP. Do not use `flash login` for this project setup.

## 1.2 Set the spending boundary

In the RunPod console:

1. Open **Billing**.
2. Add no more than **US$10 prepaid credit**.
3. Leave automatic payments disabled.
4. Do not deploy a Pod from the billing page.

The prepaid balance limits available spend, but it does not replace teardown. A stopped Pod or retained volume can continue to cost money.

## 1.3 Confirm the account is empty

Open a fresh Codex task after OAuth and ask:

> List my RunPod Pods and network volumes. Do not create, start, stop, or delete anything.

**Success looks like:** both lists are empty. Verified on 2026-10-01. Existing resources are not automatically ours; stop and inspect them before continuing.

## 1.4 Register the existing SSH public key

Run on the Mac:

```bash
test -f "$HOME/.ssh/id_ed25519_runpod.pub"
pbcopy < "$HOME/.ssh/id_ed25519_runpod.pub"
```

In RunPod, open **Settings > SSH Public Keys**, add a key, and paste it. Register the public `.pub` file only. Never upload or paste the private key at `$HOME/.ssh/id_ed25519_runpod`.

SSH keys must be registered before Pod creation because RunPod injects them when the Pod boots.

## 1.5 Phase 1 readiness check

- [x] Official RunPod plugin installed and enabled.
- [x] RunPod MCP reports OAuth authentication.
- [x] Account has no unexpected Pods or network volumes.
- [x] Prepaid balance is at most US$10 (user-confirmed).
- [x] Automatic payments are disabled (user-confirmed).
- [x] Dedicated `runpod-triton-benchmark` SSH public key is registered.

**Stop here. Do not create paid resources until ten model pairs pass Phase 2.**

---

# Phase 2 - Build the model zoo locally

This phase uses no cloud resources.

## Goal

Generate reproducible synthetic datasets and 100 model pairs. Each pair contains two representations of the same trained XGBoost regressor:

- An ONNX model for the ONNX Runtime backend.
- An XGBoost UBJ model for the FIL backend.

## Rules

- Use one fixed master random seed.
- Record the exact input and output tensor names.
- Use ONNX opset 15 for every exported model.
- Validate every generated model locally.
- Get ten complete model pairs working before scaling to 100.

## Commands

Run the ten-pair gate first:

```bash
source .venv/bin/activate
python scripts/gen_models.py --count 10 --seed 20260805
python scripts/validate_models.py --count 10
```

Only after all ten pass:

```bash
python scripts/gen_models.py --count 100 --seed 20260805
python scripts/validate_models.py --count 100
```

Generated model files, manifests, and `config.pbtxt` files are reproducible build artifacts and are ignored by Git.

**Success looks like:** the final JSON report contains `"model_pairs": 100`, `"onnx_models": 100`, `"fil_models": 100`, and `"status": "valid"`.

---

# Phase 3 - Prepare and create the RunPod environment

Do this only after Phase 2 has ten validated model pairs.

## 3.1 Read live GPU availability and price

In a fresh Codex task, ask:

> Using RunPod, list current on-demand NVIDIA L4 and L40S availability and hourly prices. Read only; do not create anything.

Choose the least expensive suitable GPU that can load all 100 models. Prefer L4; use L40S only if L4 is unavailable and its complete maximum cost is approved. Record the exact GPU, cloud type, data center, hourly price, and retrieval date in `ENVIRONMENT.md`.

Live check on 2026-10-01: one 24 GB L4 in Secure Cloud was low-stock at **US$0.49/hour**. CUDA 13.0-compatible stock was available. Do not pin a data center because the benchmark is single-node and uses no network volume; allowing all Secure Cloud locations improves the chance of obtaining the exact GPU.

Do not mix measurements from different GPU types in one result table. Hardware identity is part of the experiment.

## 3.2 Build the project-specific Pod image

The stock Triton image does not define this project's SSH and startup behavior. This repository contains a `linux/amd64` image derived from:

```text
nvcr.io/nvidia/tritonserver:25.06-py3
```

The image adds key-only SSH, pinned model-generation dependencies, and the project scripts. The workflow smoke-tests Triton, Python imports, required-key failure, and successful SSH startup before it publishes anything. It never publishes a `latest` tag.

After the current changes have been pushed:

1. Open the GitHub repository in your browser.
2. Select **Actions**.
3. Select **Publish Pod image**.
4. Select **Run workflow**.
5. Enter the intended semantic release tag (next release: `0.1.4`), then run it.
6. Wait for the workflow to finish with a green check.
7. Open the workflow summary and copy the complete `ghcr.io/...@sha256:...` reference.
8. Open the new package's **Package settings** and change its visibility to **Public**. Do not add registry credentials to RunPod.
9. Paste the digest-pinned reference into the **Project image digest** row in `ENVIRONMENT.md`.

For image `0.1.0`, the build, smoke test, and registry push succeeded in run `36811835631`. The run's final reporting command failed after publication because it initially captured Docker's stdout but not stderr. The recorded image is:

```text
ghcr.io/daetan999/triton-multimodel-bench@sha256:b49adeeea3707504067b80f360af03952dc59286e5af3771baee7610321b566c
```

Image `0.1.1` was built, smoke-tested, and published in run `36816708070`. Its final reporting command initially selected the word `digest:` rather than the following SHA; the parser now extracts the complete SHA-256 value directly. Anonymous registry access returned HTTP 200, 42 layers, and this immutable digest:

```text
ghcr.io/daetan999/triton-multimodel-bench@sha256:ad62d0e0006825b741238007ef364ed63a80a51bc4bd2b27499e8f5e1eaf5ad7
```

Corrected workflow run `36818347526` then completed green and reported the
same digest.

Image `0.1.3` added the bounded ONNX Runtime thread-pool fix. Workflow run
`36827162657` completed green, and anonymous registry access verified this
immutable digest:

```text
ghcr.io/daetan999/triton-multimodel-bench@sha256:20d4c4582fefe59e3ddd62fd30fe001db790e2bb1c92417324c345bc5f76efea
```

Image `0.1.4` added the remote `rsync` binary used by the result mirror.
Workflow run `36950090842` completed green, and anonymous registry access
verified 42 layers at this immutable digest:

```text
ghcr.io/daetan999/triton-multimodel-bench@sha256:50087af8c9182b01fd1d45c6c4c7d77fffe0c321f2e6829b49d48a0add3cccbb
```

**Success looks like:** the package is public and the intended digest above appears on its package page. Future workflow runs should also end green and print the digest in their summary.

If the build fails, do not create a Pod. Copy the failed step and its complete log into the working chat.

## 3.3 Prepare continuous result mirroring

Use a 20 GB disposable container disk, but never leave the only copy of a measured result on it. Before starting the benchmark, create a dedicated local destination and capture the Pod's SSH host key:

```bash
export POD_HOST='REPLACE_WITH_POD_IP'
export POD_PORT='REPLACE_WITH_SSH_PORT'
export RESULT_NAME='portfolio-benchmark'
mkdir -p "evidence/runpod/$RESULT_NAME"
ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "evidence/runpod/$RESULT_NAME/known_hosts"
```

Start the mirror in a second Mac terminal and keep that terminal open:

```bash
cd "/Users/dae/Documents/Codex/2026-08-05/i-w/outputs/triton-multimodel-bench"
export POD_HOST='REPLACE_WITH_POD_IP'
export POD_PORT='REPLACE_WITH_SSH_PORT'
export RESULT_NAME='portfolio-benchmark'
caffeinate -dimsu .venv/bin/python scripts/mirror_results.py \
  --host "$POD_HOST" \
  --port "$POD_PORT" \
  --identity-file "$HOME/.ssh/id_ed25519_runpod" \
  --known-hosts-file "evidence/runpod/$RESULT_NAME/known_hosts" \
  --remote-dir "/workspace/results/$RESULT_NAME" \
  --local-dir "evidence/runpod/$RESULT_NAME/results" \
  --interval-seconds 30
```

Create a canary file on the Pod and confirm it appears locally before starting any measured run. The mirror never uses `--delete`, so a later remote failure cannot erase local evidence.

## 3.4 Create the benchmark Pod

Use these fixed settings:

| Setting | Required value |
|---|---|
| Name | `triton-portfolio-benchmark` |
| Compute | 1× approved NVIDIA GPU, on-demand |
| Cloud | Secure Cloud |
| CUDA host floor | 13.0 |
| Image | Project image tag and digest from Phase 3.2 |
| Container disk | 20 GB, disposable |
| Network volume | None |
| HTTP ports | None |
| TCP ports | SSH only |
| Cost guard | Approved maximum price, two-hour ceiling, and an independently scheduled termination check |

Create no public Triton port. Triton HTTP, gRPC, and metrics stay inside the Pod on ports 8000, 8001, and 8002.

Immediately record the Pod ID, creation timestamp, hourly rate, maximum quoted cost, and scheduled termination time in `ENVIRONMENT.md` and `LOG.md`. Do not describe a remembered timestamp as a hard cutoff.

## 3.5 Verify the real environment

Connect over SSH and run:

```bash
nvidia-smi
test -x /opt/tritonserver/bin/tritonserver
git --version
python3 --version
```

Then record the immutable container digest and the complete `nvidia-smi` output. Do not infer versions from the tag.

**Success looks like:** the GPU is exactly `NVIDIA L4`, the Triton executable exists, and the project can create `/workspace/model_repository`. Record Triton 2.59.0 from its startup log during Phase 4; its CLI does not return success for `--version` or `--help`.

## 3.6 End every paid session safely

1. Confirm the continuous mirror is still reporting successful copies.
2. Run one final mirror with `--once` and validate the expected summary count locally.
3. Commit only reviewed, non-secret project artifacts.
4. **Terminate** the Pod; do not merely stop it.
5. List Pods again and verify the benchmark Pod is absent.
6. List network volumes and confirm the list is still empty.

Do not terminate while the only valid result copy is still on the Pod. The scheduled termination check is the spending backstop; continuous mirroring is the evidence backstop.

---

# Phase 4 - Triton and ten-model smoke test

Do not scale to 100 models until this phase passes.

## Required checks

- [x] Ten ONNX models report `READY`.
- [x] Ten FIL models report `READY`.
- [x] One inference request succeeds against each backend.
- [x] Tensor names match the exported model files.
- [x] The Triton container digest is recorded.
- [x] Server logs contain no hidden model-load failures in the final launch.

FIL compatibility requirements discovered on the first L4 smoke run:

- Use `output_class: false` for regression models; Triton 2.59 rejects the older `is_classifier` parameter.
- Pin XGBoost 3.0.2. XGBoost 3.4.0 writes JSON/UBJSON that Triton 25.06's embedded Treelite parser cannot load.
- Serialize `model.get_booster()` directly rather than `XGBRegressor.save_model()` so newer scikit-learn estimator metadata does not affect the serving artifact.
- Compare backend outputs after flattening. ONNX returns shape `[batch, 1]`, while FIL returns `[batch]` for this regressor; the scalar values must still satisfy `rtol=1e-5` and `atol=1e-5`.

Do not run a formal benchmark from an image that lacks these fixes. The first live Pod was hot-patched for diagnosis; publish a new immutable image before the measured run.

---

# Phase 5 - Benchmark protocol

Every measured condition follows the same order.

1. Confirm the Pod ID, GPU identity, data center, image digest, model config, maximum cost, and scheduled termination time.
2. Pin Triton and the load generator to disjoint CPU sets.
3. Verify Triton with one real request before measuring.
4. Record warm-up requests separately and exclude them from all measured metrics.
5. Confirm the load-generator CPU set is not saturated.
6. Start one-second GPU and Triton-process CPU/RSS telemetry.
7. Run the local load generator against `127.0.0.1`.
8. Save one CSV row per request.
9. Stop telemetry.
10. Repeat at least three times.
11. Randomize condition order where practical.
12. Copy results off the Pod and terminate it.
13. Append the session summary to `LOG.md`.

## Portfolio experiment matrix

| Dimension | Values |
|---|---|
| Model count | 1, 10, 100 |
| Batching | Off, 10 ms queue |
| Offered load | 200 requests/second |
| Repetitions | 3 |
| Primary metrics | Throughput, p50, p95, p99, error rate |
| Supporting metrics | GPU utilisation/memory/power, client scheduler delay/CPU, Triton process CPU/RSS |

Never compare conditions produced by different code versions without explicitly recording that difference.

The portfolio ONNX matrix contains 18 runs: 3 model counts × 2 batching policies × 1 target load × 3 repetitions. Each run uses 5 seconds of warm-up and 30 seconds of measurement. This is enough to show the model-count and batching trade-offs without claiming to be an exhaustive performance study.

## 5.1 Generate and validate the 100-pair repository on the Pod

Run over SSH:

```bash
mkdir -p /workspace/model_repository /workspace/results
python3 /opt/triton-benchmark/scripts/gen_models.py \
  --count 100 \
  --seed 20260805 \
  --repository /workspace/model_repository
python3 /opt/triton-benchmark/scripts/validate_models.py \
  --count 100 \
  --repository /workspace/model_repository
```

**Success looks like:** validation prints `"status": "valid"` with 100 model pairs.

## 5.2 Confirm CPU isolation

```bash
taskset --cpu-list 0-7 true
taskset --cpu-list 8-12 true
```

Both commands must return silently with exit code zero. Triton uses CPUs 0–7; the runner uses CPUs 8–12. If the new host has a different CPU quota, choose two disjoint sets that fit inside that quota and record them before running.

## 5.3 Run the five-second live preflight

Use the exact `0.1.4` image digest and source commit:

```bash
export BENCH_IMAGE_DIGEST='ghcr.io/daetan999/triton-multimodel-bench@sha256:50087af8c9182b01fd1d45c6c4c7d77fffe0c321f2e6829b49d48a0add3cccbb'
export BENCH_GIT_COMMIT='0c40a1fda6bc6bd2989f07d8daada8906fdd3efc'
taskset --cpu-list 8-12 python3 /opt/triton-benchmark/scripts/run_matrix.py \
  --repository /workspace/model_repository \
  --results-dir /workspace/results/preflight \
  --backend onnx \
  --model-counts 100 \
  --queue-delays-us off \
  --target-qps 50 \
  --repetitions 1 \
  --warmup-seconds 2 \
  --duration-seconds 5 \
  --server-cpus 0-7 \
  --loadgen-cpus 8-12 \
  --image-digest "$BENCH_IMAGE_DIGEST" \
  --git-commit "$BENCH_GIT_COMMIT"
```

**Success looks like:** the final report says one run completed and the preflight directory contains raw request CSV, GPU CSV, Triton CPU/RSS CSV, JSON summary, matrix manifest, and server log. The JSON summary must say `"status": "valid"` and `"failed_requests": 0`.

## 5.4 Start the compact ONNX matrix

First complete the canary mirror check from Phase 3.3. Do not launch the matrix until the canary exists on the Mac.

Run it detached so an SSH disconnect does not stop the experiment:

```bash
mkdir -p /workspace/results/portfolio-benchmark
setsid taskset --cpu-list 8-12 \
  python3 /opt/triton-benchmark/scripts/run_matrix.py \
  --repository /workspace/model_repository \
  --results-dir /workspace/results/portfolio-benchmark \
  --backend onnx \
  --model-counts 1,10,100 \
  --queue-delays-us off,10000 \
  --target-qps 200 \
  --repetitions 3 \
  --warmup-seconds 5 \
  --duration-seconds 30 \
  --base-seed 20260805 \
  --server-cpus 0-7 \
  --loadgen-cpus 8-12 \
  --image-digest "$BENCH_IMAGE_DIGEST" \
  --git-commit "$BENCH_GIT_COMMIT" \
  > /workspace/results/portfolio-benchmark/runner.log 2>&1 \
  < /dev/null &
echo $! > /workspace/results/portfolio-benchmark/runner.pid
```

Monitor without altering the run:

```bash
tail -f /workspace/results/portfolio-benchmark/runner.log
```

The runner randomizes server-configuration groups and the runs inside each group deterministically. It refuses to overwrite artifacts. If the process ends between runs, repeat the command with `--resume`. If it stops during a run, preserve the partial files for diagnosis and use a new results directory; do not delete evidence to force a resume.

## 5.5 Validate and copy results before teardown

```bash
test "$(find /workspace/results/portfolio-benchmark/summaries -name '*.json' | wc -l)" -eq 18
grep -R '"status": "invalid"' /workspace/results/portfolio-benchmark/summaries && exit 1 || true
find /workspace/results/portfolio-benchmark -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > /workspace/results/portfolio-benchmark/SHA256SUMS
```

Run the mirror once more with `--once`, confirm all 18 summaries and `SHA256SUMS` exist on the Mac, and only then terminate the Pod.

---

# Phase 6 - Publication gate

Do not make a performance claim until all boxes are checked.

- [ ] Every chart traces back to committed raw CSV files.
- [ ] Every run records its configuration and repetition number.
- [ ] Warm-up requests are excluded and the exclusion is documented.
- [ ] Failed and partial runs remain visible but are marked invalid.
- [ ] Triton and the load generator used recorded, disjoint CPU affinities.
- [ ] The load generator was not CPU-saturated.
- [ ] The single-node load-generation limitation is stated prominently.
- [ ] Prices include provider, region, currency, and retrieval date.
- [ ] The README calls this a synthetic analogue.
- [ ] Results are phrased as applying to this workload, not all Triton workloads.
- [ ] The GPU and CPU crossover point is published, including where GPU loses.

---

# Emergency cost check

If you are unsure whether anything is still running:

1. Open the RunPod **Pods** page.
2. If the benchmark Pod exists, click **Terminate** and confirm.
3. Confirm the Pods list is empty.
4. Open **Storage** and confirm no network volume was created.

Stopping is not the emergency action: terminate the Pod so retained Pod storage cannot continue billing.

# End-of-session log template

Append this block to `LOG.md`:

```markdown
## YYYY-MM-DD - Session N

### Goal

### Commands run

### What worked

### What failed

### Versions and configuration

### Files produced

### Next step
```
