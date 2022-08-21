#!/usr/bin/env bash
set -x

BOOTSTRAP_PORT=8998

python3 -u -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --served-model-name deepseek-ai/DeepSeek-V2-Lite-Chat \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 7000 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
  --dp-size $DP_SIZE \
  --tp-size 1 \
  --nnodes 1 \
  --node-rank 0 \
  --enable-mixed-chunk \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --prefill-round-robin-balance
