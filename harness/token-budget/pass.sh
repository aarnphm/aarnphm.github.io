#!/bin/sh
# Stages 0-4 of a token budget pass: inventory, briefs, read-only proposals from Sol and Argon,
# validation, and one agent-decide verdict per proposal. Stage 5 onward (worktree, move, deepen,
# land) stays with the agent that reads the results (.agents/skills/token-budget/SKILL.md).
#   harness/token-budget/pass.sh [--top N] [--run <dir>] [<path>...]
# With paths it plans those files; without, the --top N (default 3) by tokens x churn.
# TB_WORKERS (default "sol argon"; empty runs only the baseline) picks the workers. With both
# workers a pass takes 10-20 minutes.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
WS=$(git -C "$HERE" rev-parse --show-toplevel)
TOP=3
RUN=
while [ $# -gt 0 ]; do
  case $1 in
    --top) TOP=$2; shift 2 ;;
    --run) RUN=$2; shift 2 ;;
    --) shift; break ;;
    -*) echo "usage: $0 [--top N] [--run <dir>] [<path>...]" >&2; exit 2 ;;
    *) break ;;
  esac
done
RUN=${RUN:-$WS/work/refactor/pass-$(date +%Y-%m-%d-%H%M)}
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
cd "$WS"

# 0-1: inventory from the index, graph and briefs.
if [ $# -gt 0 ]; then
  "$HERE/check.py" --paths "$@" --json > "$RUN/inventory.json"
  "$HERE/inventory.py" "$RUN/inventory.json" --paths "$@" --out "$RUN/graph.json" --briefs "$RUN"
else
  "$HERE/check.py" --all --json > "$RUN/inventory.json"
  "$HERE/inventory.py" "$RUN/inventory.json" --top "$TOP" --out "$RUN/graph.json" --briefs "$RUN"
fi
git status --short > "$RUN/status-before.txt"
targets=$(jq -r '.files[].path' "$RUN/graph.json")
[ -n "$targets" ] || { echo "no over-budget targets" >&2; exit 1; }
echo "run: $RUN" >&2
echo "$targets" | sed 's/^/target: /' >&2

# 2: proposals, all workers and targets in parallel.
for w in ${TB_WORKERS-sol argon}; do "$HERE/dispatch.sh" "$RUN" "$w" & done
wait

# 3: proposals, baseline, validation, decision requests.
"$HERE/collect.sh" "$RUN" || true

# 4: one decision per proposal (assessor none, so this costs nothing).
for req in "$RUN"/*/decisions/*.decision-request.json; do
  [ -f "$req" ] || continue
  out=${req%.decision-request.json}
  agent-decide < "$req" > "$out.decision.json" 2> "$out.decision.stderr.txt" \
    || echo "agent-decide failed: $req" >&2
done

# Workers are read-only; a changed target blob means someone else edited it during the run.
for t in $targets; do
  [ "$(git rev-parse ":$t")" = "$(jq -r --arg t "$t" '.files[] | select(.path == $t) | .blob' "$RUN/graph.json")" ] \
    || echo "warning: $t changed in the index during the run; re-run before acting" >&2
done

for v in "$RUN"/*/validation.json; do
  [ -f "$v" ] || continue
  s=$(dirname "$v")
  echo
  jq -r '.target' "$v"
  jq -r '.results[] | [.worker, .status, (.metrics | "files=\(.files) largest=\(.largest) exports \(.exportsBefore)->\(.exportsAfter) cutShare=\(.cutShare) sharedState=\(.sharedState)")] | @tsv' "$v" \
    | while IFS="$(printf '\t')" read -r worker status metrics; do
        d="$s/decisions/$(printf '%s' "$worker" | sed 's/[^A-Za-z0-9_.-][^A-Za-z0-9_.-]*/-/g').decision.json"
        action=$(jq -r '.action // "?"' "$d" 2> /dev/null || echo '?')
        printf '  %-28s %-5s decide=%-9s %s\n' "$worker" "$status" "$action" "$metrics"
      done
  jq -r '.agreement[] | "  ARI \(.a) vs \(.b): \(.ari)"' "$v"
done
echo
echo "results: $RUN/<target>/validation.json and decisions/" >&2
