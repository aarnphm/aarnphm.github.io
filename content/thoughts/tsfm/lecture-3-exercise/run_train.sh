#!/usr/bin/env bash
set -euo pipefail

python -m minigpt.np.train --steps 200 --d_model 128 --n_heads 4 --n_layers 2 --seq 128 --batch 16 --lr 3e-4
