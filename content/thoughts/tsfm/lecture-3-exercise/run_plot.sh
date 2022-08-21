#!/usr/bin/env bash
set -euo pipefail

# ASCII plot for latest checkpoint losses.
# Usage:
#   ./run_plot.sh                          # plot ${SCRIPT_DIR}/checkpoints
#   ./run_plot.sh /path/to/ckpts           # choose a checkpoints dir
#   ./run_plot.sh --width 120 --height 20  # pass args through
#   CKPT_DIR=... PLOT_WIDTH=120 PLOT_HEIGHT=20 ./run_plot.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

# Detect if first arg is a path (doesn't start with '-') and use as CKPT_DIR
if [[ ${1-}:- != -* && ${1-} != :- ]]; then
  CKPT_DIR_INPUT="$1"
  shift || true
else
  CKPT_DIR_INPUT=""
fi

CKPT_DIR="${CKPT_DIR_INPUT:-${CKPT_DIR:-"${SCRIPT_DIR}/checkpoints"}}"
WIDTH="${PLOT_WIDTH:-}" 
HEIGHT="${PLOT_HEIGHT:-}"

CMD=(python -u -m minigpt.np.ascii_losses --ckpt-dir "${CKPT_DIR}")
if [[ -n "${WIDTH}" ]]; then CMD+=(--width "${WIDTH}"); fi
if [[ -n "${HEIGHT}" ]]; then CMD+=(--height "${HEIGHT}"); fi

# Pass through any extra CLI flags
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run_plot] ${CMD[*]}"
exec "${CMD[@]}"

