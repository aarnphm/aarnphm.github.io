#!/usr/bin/env bash
set -x

kubectl run sgl-dev \
  --image=docker.io/rocm/sgl-dev:v0.5.3rc0-rocm630-mi30x-20251002 \
  --rm -it --restart=Never \
  --overrides="$(
    cat <<EOF
{
  "spec": {
    "hostNetwork": true,
    "hostIPC": true,
    "containers": [{
      "name": "sgl-dev",
      "image": "docker.io/rocm/sgl-dev:v0.5.3rc0-rocm630-mi30x-20251002",
      "stdin": true,
      "tty": true,
      "command": ["/bin/bash"],
      "args": ["-c", "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 && export HF_HUB_CACHE_DIR=/root/.cache/huggingface/hub && pip install bentoml && cd /workspace && bentoml serve --debug --arg num_prefill=4 --arg num_decode=4"],
      "securityContext": {
        "privileged": true,
        "capabilities": {
          "add": ["SYS_ADMIN", "SYS_PTRACE"]
        },
        "seccompProfile": {
          "type": "Unconfined"
        }
      },
      "volumeMounts": [
        {
          "name": "workspace",
          "mountPath": "/workspace"
        },
        {
          "name": "hf-cache",
          "mountPath": "/root/.cache"
        },
        {
          "name": "dshm",
          "mountPath": "/dev/shm"
        },
        {
          "name": "dev-kfd",
          "mountPath": "/dev/kfd"
        },
        {
          "name": "dev-dri",
          "mountPath": "/dev/dri"
        }
      ],
      "resources": {
        "limits": {
          "amd.com/gpu": 8
        }
      }
    }],
    "volumes": [
      {
        "name": "workspace",
        "hostPath": {
          "path": "$(pwd)",
          "type": "Directory"
        }
      },
      {
        "name": "hf-cache",
        "hostPath": {
          "path": "/root/.cache",
          "type": "Directory"
        }
      },
      {
        "name": "dshm",
        "emptyDir": {
          "medium": "Memory",
          "sizeLimit": "32Gi"
        }
      },
      {
        "name": "dev-kfd",
        "hostPath": {
          "path": "/dev/kfd"
        }
      },
      {
        "name": "dev-dri",
        "hostPath": {
          "path": "/dev/dri"
        }
      }
    ]
  }
}
EOF
  )"
