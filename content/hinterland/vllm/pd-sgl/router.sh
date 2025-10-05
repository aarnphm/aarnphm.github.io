#!/usr/bin/env bash
set -x

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:6000" \
  --decode "http://127.0.0.1:7000" \
  --host 0.0.0.0 \
  --port 8000
