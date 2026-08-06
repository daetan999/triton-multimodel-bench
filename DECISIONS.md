# Decision record

## D001 - Use NVIDIA L4 rather than T4

**Status:** Accepted
**Date:** 2026-08-05

Use Google Cloud `g2-standard-4`: one NVIDIA L4, four vCPUs, 16 GB system memory, and 24 GB GPU memory.

The L4 is a current inference-focused data-centre GPU and is available in Singapore. The project is a modern synthetic analogue of the earlier architecture, not a hardware-identical reproduction.

## D002 - Use a separate load generator

**Status:** Accepted
**Date:** 2026-08-05

Use an `n2-standard-4` VM in the same zone to generate traffic. Reusing it for the CPU baseline limits cost while preventing the load generator from consuming the GPU server's CPU resources.

## D003 - Pin Triton 25.06

**Status:** Accepted
**Date:** 2026-08-05

Use `nvcr.io/nvidia/tritonserver:25.06-py3`. It contains Triton 2.59.0, CUDA 12.9.1, ONNX Runtime 1.22.0, and FIL. This matches the selected Google Deep Learning VM's CUDA 12.9 generation and NVIDIA 580 driver more conservatively than a CUDA 13 container.

## D004 - Never expose Triton publicly

**Status:** Accepted
**Date:** 2026-08-05

Triton ports 8000, 8001, and 8002 remain private. The load generator connects using the VM's internal IP address. No public firewall rule will be created for these ports.

## D005 - Pin ONNX opset 15

**Status:** Accepted
**Date:** 2026-08-05

Use ONNX opset 15 for both scikit-learn and LightGBM exports. The installed LightGBM converter rejects higher target opsets; one shared opset prevents the model zoo from mixing formats.
