# Triton Multi-Model Benchmark Runbook

This is the operating guide for the whole project. Work from top to bottom. Do not skip a verification box.

## Where we are now

- [x] Repository created and cloned.
- [x] Google Cloud CLI installed.
- [x] Python 3.12 environment created.
- [x] Pinned packages installed and checked.
- [x] scikit-learn and LightGBM ONNX export tests passing.
- [ ] Google Cloud authentication completed.
- [ ] Billing project and budget configured.
- [ ] NVIDIA L4 quota approved.

Your next unfinished step is **Phase 1.1 - Sign in**.

## How to use this runbook

- Run one command block at a time.
- Unless a section says otherwise, run local commands from the repository root.
- Compare your output with **Success looks like** before continuing.
- If a command fails, copy the complete command and complete error into the working chat.
- Never substitute invented benchmark values for missing measurements.
- Never run a paid VM without a runtime limit.

## Fixed project choices

| Item | Choice |
|---|---|
| Cloud | Google Cloud |
| Region | Singapore, `asia-southeast1` |
| Preferred zone | `asia-southeast1-a` |
| GPU server | `g2-standard-4`, one NVIDIA L4 |
| Load generator / CPU baseline | `n2-standard-4` |
| GPU OS | Google Deep Learning VM, Ubuntu 22.04, CUDA 12.9, NVIDIA 580 |
| Triton container | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| ONNX opset | 15 |
| Maximum VM run | Six hours per start |
| Maximum project budget | S$35 |

---

# Phase 0 - Local setup

Codex has already installed Google Cloud CLI and created the repository scaffold.

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
pytest -q
```

**Success looks like:** `Local Python environment: READY`, `No broken requirements found`, and `2 passed`.

## 0.3 Confirm repository state

```bash
git status --short
```

**Success looks like:** the new scaffold files appear. Do not commit or push yet.

---

# Phase 1 - Google Cloud account and quota

No GPU is created in this phase. This phase should cost nothing.

## 1.1 Sign in

```bash
gcloud auth login
```

A browser opens. Sign in with the Google account that owns the billing account.

Verify:

```bash
gcloud auth list --filter=status:ACTIVE --format='value(account)'
```

**Success looks like:** your Google account email appears once.

## 1.2 Choose a globally unique project ID

Start with this value:

```bash
export TRITON_PROJECT_ID="dae-triton-bench-2026"
```

Create the project:

```bash
gcloud projects create "$TRITON_PROJECT_ID" \
  --name="Triton Multi-Model Benchmark"
```

If Google says the ID is already in use, append four digits to the value and run the create command again.

Select it:

```bash
gcloud config set project "$TRITON_PROJECT_ID"
gcloud config set compute/region asia-southeast1
gcloud config set compute/zone asia-southeast1-a
```

Verify:

```bash
gcloud config list --format='text(core.project,compute.region,compute.zone)'
```

**Success looks like:** your project ID, `asia-southeast1`, and `asia-southeast1-a` all appear.

## 1.3 Attach billing

List the billing accounts you can use:

```bash
gcloud billing accounts list
```

Copy the billing account ID from the `ACCOUNT_ID` column, then set it:

```bash
export TRITON_BILLING_ACCOUNT="PASTE_ACCOUNT_ID_HERE"
gcloud billing projects link "$TRITON_PROJECT_ID" \
  --billing-account="$TRITON_BILLING_ACCOUNT"
```

Verify:

```bash
gcloud billing projects describe "$TRITON_PROJECT_ID" \
  --format='value(billingEnabled)'
```

**Success looks like:** `True`.

## 1.4 Enable the required APIs

```bash
gcloud services enable \
  compute.googleapis.com \
  cloudbilling.googleapis.com \
  serviceusage.googleapis.com
```

Verify:

```bash
gcloud services list --enabled \
  --filter='name:compute.googleapis.com' \
  --format='value(name)'
```

**Success looks like:** `compute.googleapis.com`.

## 1.5 Set the budget alert

Open the Google Cloud console:

1. Go to **Billing > Budgets & alerts**.
2. Create a budget named `triton-benchmark-safety`.
3. Scope it to this project only.
4. Set the amount to **S$35**, or the equivalent in your billing currency.
5. Keep alerts at 50%, 80%, and 100%.

Important: a normal budget sends alerts; it does not stop Compute Engine. Our VM runtime limit is the actual safety control.

## 1.6 Request GPU quota

In Google Cloud Console:

1. Open **IAM & Admin > Quotas & System Limits**.
2. Filter service to **Compute Engine API**.
3. Request a limit of `1` for **NVIDIA L4 GPUs** in `asia-southeast1`.
4. Request a limit of `1` for **GPUs (all regions)** if the current limit is zero.
5. Submit the request and wait for approval.

Do not create a substitute GPU while approval is pending.

## 1.7 Phase 1 readiness check

```bash
gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud config get-value project
gcloud billing projects describe "$TRITON_PROJECT_ID" \
  --format='value(billingEnabled)'
```

Check all three:

- [ ] Account email appears.
- [ ] Correct project ID appears.
- [ ] Billing prints `True`.
- [ ] L4 quota request is approved or pending.
- [ ] S$35 budget alert exists.

**Stop here until the quota is approved.** Continue with model generation while waiting.

---

# Phase 2 - Build the model zoo locally

This phase uses no cloud resources.

## Goal

Generate reproducible synthetic datasets and 100 model pairs:

- An ONNX model for the ONNX Runtime backend.
- An XGBoost model for the FIL backend.

## Rules

- Use one fixed master random seed.
- Record the exact input and output tensor names.
- Use ONNX opset 15 for every exported model.
- Validate every generated model locally.
- Get ten complete model pairs working before scaling to 100.

## Commands

The generation and validation scripts will be added in the next implementation step. When present, the commands will be:

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

**Success looks like:** validation reports exactly 100 ONNX models and 100 FIL-compatible models with no failures.

---

# Phase 3 - Create the cloud machines

Do this only after Phase 2 has ten validated model pairs and L4 quota is approved.

## 3.1 Resolve and record the exact GPU image

```bash
export TRITON_IMAGE="$(gcloud compute images describe-from-family \
  common-cu129-ubuntu-2204-nvidia-580 \
  --project=deeplearning-platform-release \
  --format='value(name)')"
echo "$TRITON_IMAGE"
```

Copy the result into `ENVIRONMENT.md` before creating the VM.

## 3.2 Create the L4 server

```bash
gcloud compute instances create triton-l4 \
  --zone=asia-southeast1-a \
  --machine-type=g2-standard-4 \
  --image="$TRITON_IMAGE" \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --max-run-duration=6h \
  --instance-termination-action=STOP \
  --labels=project=triton-benchmark,role=server
```

Do not add firewall rules for ports 8000, 8001, or 8002.

## 3.3 Create the load-generator and CPU-baseline VM

```bash
gcloud compute instances create triton-loadgen \
  --zone=asia-southeast1-a \
  --machine-type=n2-standard-4 \
  --image-family=ubuntu-2204-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --max-run-duration=6h \
  --instance-termination-action=STOP \
  --labels=project=triton-benchmark,role=loadgen
```

## 3.4 Verify the L4

```bash
gcloud compute ssh triton-l4 --zone=asia-southeast1-a
```

On the VM:

```bash
nvidia-smi
docker --version
```

**Success looks like:** `NVIDIA L4` appears and Docker reports a version.

## 3.5 Pull and record Triton

On the L4 VM:

```bash
docker pull nvcr.io/nvidia/tritonserver:25.06-py3
docker image inspect \
  --format='{{index .RepoDigests 0}}' \
  nvcr.io/nvidia/tritonserver:25.06-py3
```

Copy the digest into `ENVIRONMENT.md`.

Verify GPU access:

```bash
docker run --rm --gpus=all \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  nvidia-smi
```

**Success looks like:** the container also reports `NVIDIA L4`.

## 3.6 End every paid session safely

Run from your Mac:

```bash
gcloud compute instances stop triton-l4 triton-loadgen \
  --zone=asia-southeast1-a
```

Verify:

```bash
gcloud compute instances list \
  --filter='name=(triton-l4 triton-loadgen)' \
  --format='table(name,status)'
```

**Success looks like:** both machines show `TERMINATED`. In Google Cloud, `TERMINATED` means stopped, not deleted.

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

1. Start both VMs.
2. Confirm the VM names, zone, image, container digest, and model config.
3. Warm up without recording results.
4. Start GPU telemetry.
5. Run the load generator from `triton-loadgen`.
6. Save one CSV row per request.
7. Stop GPU telemetry.
8. Repeat at least three times.
9. Stop both VMs.
10. Append the session summary to `LOG.md`.

## Experiment matrix

| Dimension | Values |
|---|---|
| Model count | 1, 10, 100 |
| Batching | Off, 2 ms queue, 10 ms queue |
| Offered load | Low, medium, high; fixed after calibration |
| Repetitions | At least 3 |
| Primary metrics | Throughput, p50, p95, p99, error rate |
| Supporting metrics | GPU utilisation, GPU memory, server queue time |

Never compare conditions produced by different code versions without explicitly recording that difference.

---

# Phase 6 - Publication gate

Do not make a performance claim until all boxes are checked.

- [ ] Every chart traces back to committed raw CSV files.
- [ ] Every run records its configuration and repetition number.
- [ ] Warm-up requests are excluded and the exclusion is documented.
- [ ] Failed and partial runs remain visible but are marked invalid.
- [ ] A separate load generator was used.
- [ ] The load generator was not CPU-saturated.
- [ ] Prices include provider, region, currency, and retrieval date.
- [ ] The README calls this a synthetic analogue.
- [ ] Results are phrased as applying to this workload, not all Triton workloads.
- [ ] The GPU and CPU crossover point is published, including where GPU loses.

---

# Emergency cost check

If you are unsure whether anything is still running:

```bash
gcloud compute instances list \
  --filter='status=RUNNING' \
  --format='table(name,zone,machineType,status)'
```

If either project VM appears, stop it immediately:

```bash
gcloud compute instances stop triton-l4 triton-loadgen \
  --zone=asia-southeast1-a --quiet
```

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
