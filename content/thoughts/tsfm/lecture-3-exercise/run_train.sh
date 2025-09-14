#!/usr/bin/env bash
set -euo pipefail

python -m minigpt.np.train --seq 256 --d_model 128 --n_heads 4 --n_layers 2 --batch 16 --prefetch 8
