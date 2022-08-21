#!/usr/bin/env bash
set -x

IMAGE="rocm/sgl-dev:v0.5.3rc0-rocm630-mi30x-20251002"

HF_CACHE_HOST="/root/.cache"
HF_CACHE_SUBDIR="huggingface/hub"

nerdctl run -it --rm \
  --privileged \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --group-add=video \
  --ipc=host \
  --device=/dev/infiniband/rdma_cm \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  --mount type=bind,src=$(pwd),dst=/workspace,rw \
  --mount type=bind,src=${HF_CACHE_HOST},dst=/root/.cache,rw \
  ${IMAGE} \
  /bin/bash -c "export HIP_VISIBLE_DEVICES=4,5,6,7 && \
                export HF_HUB_CACHE_DIR=/root/.cache/${HF_CACHE_SUBDIR} && \
                export DP_SIZE=4 && \
                cd /workspace && \
                bash ./decode.sh"
