#!/usr/bin/env bash
set -euo pipefail

python -m minigpt.np.inference --top_k 32 --prompt "The meaning of life is" --max_new_tokens 32
