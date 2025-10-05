#!/usr/bin/env bash
set -x

# Bootstrap port for disaggregation (prefill listens)
BOOTSTRAP_PORT=8998

UCX_NET_DEVICES=mlx5_0 python3 -u -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 6000 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
  --disaggregation-ib-device mlx5_0 \
  --dp-size $DP_SIZE \
  --tp-size 1 \
  --nnodes 1 \
  --node-rank 0 \
  --enable-mixed-chunk \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --load-balance-method round_robin
