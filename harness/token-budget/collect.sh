#!/bin/sh
# Stages 2-3: extract each finished worker's proposal, add the contiguous-cut baseline, validate
# against the graph and write agent-decide requests.
#   harness/token-budget/collect.sh <run-dir>
# Writes <slug>/<worker>/proposal.json, <slug>/validation.json and <slug>/decisions/.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WS=$(git -C "$HERE" rev-parse --show-toplevel)
[ $# -eq 1 ] || { echo "usage: $0 <run-dir>" >&2; exit 2; }
RUN=$(cd "$1" && pwd)
cd "$WS" || exit 1
status=0

for s in "$RUN"/*/brief.md; do
  s=$(dirname "$s")
  target=$(jq -r --arg slug "$(basename "$s")" \
    '.files[] | select((.path | gsub("/"; "-")) == $slug) | .path' "$RUN/graph.json")
  mkdir -p "$s/baseline"
  "$HERE/baseline.py" "$RUN/graph.json" "$target" > "$s/baseline/proposal.json"
  for d in "$s"/*/; do
    d=${d%/}
    # Worker folders are one word (sol, argon); archived runs carry a suffix.
    case $(basename "$d") in *[!a-z]*) continue ;; esac
    [ -f "$d/exit-code" ] || continue
    if [ -f "$d/last-message.txt" ]; then
      text=$(cat "$d/last-message.txt")
    else
      python3 "$HOME/.claude/skills/argon/scripts/inspect_run.py" "$d/events.ndjson" \
        --exit-code "$(cat "$d/exit-code")" --stderr "$d/stderr.txt" > "$d/inspect.json"
      text=$(jq -r '.response // .result // .final_response // ""' "$d/inspect.json")
    fi
    printf '%s' "$text" | python3 -c '
import json, re, sys
blocks = re.findall(r"```json\s*\n(.*?)```", sys.stdin.read(), re.S)
if not blocks:
  sys.exit(3)
print(json.dumps(json.loads(blocks[-1]), indent=2, ensure_ascii=False))
' > "$d/proposal.json.tmp" && mv "$d/proposal.json.tmp" "$d/proposal.json" \
      || { rm -f "$d/proposal.json.tmp"; echo "$d: no proposal in final message" >&2; }
  done
  "$HERE/validate.py" "$RUN/graph.json" "$s"/*/proposal.json \
    --out "$s/validation.json" --decisions "$s/decisions" > /dev/null || status=1
done
exit $status
