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

The next user-only step is **Phase 3.2 - Make the GHCR package public**. Do not create a Pod yet.

## How to use this runbook

- Run one command block at a time.
- Unless a section says otherwise, run local commands from the repository root.
- Compare your output with **Success looks like** before continuing.
- If a command fails, copy the complete command and complete error into the working chat.
- Never substitute invented benchmark values for missing measurements.
- Never create a paid Pod without a six-hour termination deadline.
- Terminate Pods when a session ends; stopping a Pod can leave storage charges running.

## Fixed project choices

| Item | Choice |
|---|---|
| Cloud | RunPod, on-demand; Secure Cloud preferred |
| Data center | Chosen immediately before launch from live L4 availability |
| GPU server | One NVIDIA L4 with 24 GB VRAM |
| Load generator | Same Pod, pinned to reserved CPU cores |
| CPU baseline | Same Pod CPU, with GPU execution disabled |
| Persistent storage | 20 GB network volume mounted at `/workspace` |
| Triton container | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| ONNX opset | 15 |
| Maximum Pod run | Six hours per creation |
| Maximum project funding | US$10 prepaid credit; automatic payments off |

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

## 3.1 Read live L4 availability and price

In a fresh Codex task, ask:

> Using RunPod, list current on-demand NVIDIA L4 availability and hourly prices by cloud and data center. Read only; do not create anything.

Choose one L4. Prefer Secure Cloud when available at a reasonable price. Record the exact GPU name, cloud type, data center, hourly price, and retrieval date in `ENVIRONMENT.md`.

Do not substitute an RTX 4090 or another GPU without adding a new decision record. Hardware identity is part of the experiment.

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
5. Leave the image tag as `0.1.0`, then run it.
6. Wait for the workflow to finish with a green check.
7. Open the workflow summary and copy the complete `ghcr.io/...@sha256:...` reference.
8. Open the new package's **Package settings** and change its visibility to **Public**. Do not add registry credentials to RunPod.
9. Paste the digest-pinned reference into the **Project image digest** row in `ENVIRONMENT.md`.

For image `0.1.0`, the build, smoke test, and registry push succeeded in run `36811835631`. The run's final reporting command failed after publication because it initially captured Docker's stdout but not stderr. The recorded image is:

```text
ghcr.io/daetan999/triton-multimodel-bench@sha256:b49adeeea3707504067b80f360af03952dc59286e5af3771baee7610321b566c
```

**Success looks like:** the package is public and the digest above appears on its package page. Future workflow runs should also end green and print the digest in their summary.

If the build fails, do not create a Pod. Copy the failed step and its complete log into the working chat.

## 3.3 Create persistent storage first

Create one **20 GB standard network volume** in the selected data center. Name it `triton-benchmark`. It will mount at `/workspace` and hold the checked-out repository, logs, and raw results.

Record the volume ID in `ENVIRONMENT.md`. Do not store credentials in the volume or repository.

## 3.4 Create the L4 Pod

Use these fixed settings:

| Setting | Required value |
|---|---|
| Name | `triton-l4` |
| Compute | 1× NVIDIA L4, on-demand |
| Cloud | Secure preferred; record actual value |
| Image | Project image tag and digest from Phase 3.2 |
| Container disk | 30 GB |
| Network volume | `triton-benchmark`, mounted at `/workspace` |
| HTTP ports | None |
| TCP ports | SSH only |
| Automatic guard | Terminate after six hours |

Create no public Triton port. Triton HTTP, gRPC, and metrics stay inside the Pod on ports 8000, 8001, and 8002.

Immediately record the Pod ID, creation timestamp, and automatic termination timestamp in `ENVIRONMENT.md` and `LOG.md`.

## 3.5 Verify the real environment

Connect over SSH and run:

```bash
nvidia-smi
tritonserver --version
git --version
python3 --version
```

Then record the immutable container digest and the complete `nvidia-smi` output. Do not infer versions from the tag.

**Success looks like:** the GPU is exactly `NVIDIA L4`, Triton reports 2.59.0, and the repository volume is mounted at `/workspace`.

## 3.6 End every paid session safely

1. Copy all new code, raw results, telemetry, and logs off the Pod.
2. Commit only reviewed, non-secret project artifacts.
3. **Terminate** the Pod; do not merely stop it.
4. List Pods again and verify `triton-l4` is absent.
5. Keep the network volume only while another paid session is planned; otherwise delete it too.

The six-hour deadline is only a backstop. Manual termination is still required at the end of each session.

---

# Phase 4 - Triton and ten-model smoke test

Do not scale to 100 models until this phase passes.

## Required checks

- [ ] Ten ONNX models report `READY`.
- [ ] Ten FIL models report `READY`.
- [ ] One inference request succeeds against each backend.
- [ ] Tensor names match the exported model files.
- [ ] The Triton container digest is recorded.
- [ ] Server logs contain no hidden model-load failures.

The exact model configuration and launch commands will be added after the local model zoo fixes the real tensor names and shapes.

---

# Phase 5 - Benchmark protocol

Every measured condition follows the same order.

1. Confirm the Pod ID, GPU identity, data center, image digest, model config, and termination deadline.
2. Pin Triton and the load generator to disjoint CPU sets.
3. Verify Triton with one real request before measuring.
4. Warm up without recording results.
5. Confirm the load-generator CPU set is not saturated.
6. Start GPU and CPU telemetry.
7. Run the local load generator against `127.0.0.1`.
8. Save one CSV row per request.
9. Stop telemetry.
10. Repeat at least three times.
11. Randomize condition order where practical.
12. Copy results off the Pod and terminate it.
13. Append the session summary to `LOG.md`.

## Experiment matrix

| Dimension | Values |
|---|---|
| Model count | 1, 10, 100 |
| Batching | Off, 2 ms queue, 10 ms queue |
| Offered load | Low, medium, high; fixed after calibration |
| Repetitions | At least 3 |
| Primary metrics | Throughput, p50, p95, p99, error rate |
| Supporting metrics | GPU utilisation, GPU memory, CPU utilisation, server queue time |

Never compare conditions produced by different code versions without explicitly recording that difference.

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
2. If `triton-l4` exists, click **Terminate** and confirm.
3. Open **Storage** and inspect `triton-benchmark`.
4. Delete the network volume if no further paid session is planned.
5. Confirm the Pods list is empty.

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
