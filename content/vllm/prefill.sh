#!/usr/bin/env bash
set -x

# Bootstrap port for disaggregation (prefill listens)
BOOTSTRAP_PORT=8998

HIP_VISIBLE_DEVICES=4,5 UCX_NET_DEVICES=mlx5_0 python3 -u -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 6000 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
  --disaggregation-ib-device bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,b nxt_re7,bnxt_re8 \
  --dp-size 4 \
  --tp-size 1 \
  --nnodes 1 \
  --node-rank 0 \
  --enable-mixed-chunk \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --load-balance-method round_robin
