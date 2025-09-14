#!/usr/bin/env bash
set -euo pipefail

# Run training in the background and log output to a file so it can be tailed.
# Accepts optional extra args to pass through to the trainer.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
LOG_DIR="${LOG_DIR:-"${SCRIPT_DIR}/logs"}"
mkdir -p "${LOG_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${ts}.log"
PID_FILE="${LOG_DIR}/train.pid"

CMD=(python -u -m minigpt.np.train \
  --seq 256 --d_model 128 --n_heads 4 --n_layers 2 \
  --batch 16 --prefetch 8
)

# Allow passing extra args, e.g.: ./run_train.sh --steps 10000 --name run1
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run_train] Starting: ${CMD[*]}" | tee -a "${LOG_FILE}"
echo "[run_train] Logs: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Disable tqdm in non-interactive logging unless explicitly forced
export NO_TQDM="${NO_TQDM:-1}"

nohup "${CMD[@]}" >>"${LOG_FILE}" 2>&1 &
pid=$!
echo "$pid" >"${PID_FILE}"

echo "[run_train] Started training (PID ${pid})." | tee -a "${LOG_FILE}"
echo "[run_train] Tail logs with: tail -f ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "[run_train] Stop with: kill \$(cat ${PID_FILE})" | tee -a "${LOG_FILE}"
