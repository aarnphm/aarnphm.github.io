#!/bin/sh
# One hill-climb worker: its worktree and watcher, a 5-rep baseline, the assignment, the model run,
# then the watcher down. Output under <run-dir>/<name>/.
#   harness/base-perf/climb.sh <run-dir> <name> <group> <port> <sol|astra|fable|argon|caw>
# Workers: sol = codex exec gpt-6.1-sol, astra = codex exec gpt-6-astra (both at BP_CODEX_EFFORT,
# medium, workspace-write sandbox); fable = claude --bare on the API credits in .env
# (CLAUDE_API_KEY), claude-fable-5-1 at BP_FABLE_EFFORT (medium); argon = Gemini 4 Argon through
# the local argon launcher; caw = Cursor agent, model BP_CAW_MODEL (grok-4.7-medium).
# BP_MAX_EXPERIMENTS (8) bounds the loop; BP_BUDGET_USD (60) caps a fable run.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
WS=$(git -C "$HERE" rev-parse --show-toplevel)
[ $# -eq 5 ] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
RUN=$1; NAME=$2; GROUP=$3; PORT=$4; WORKER=$5
# The harness is the main checkout's, by absolute path: a worker's branch may predate it and the
# worker must never edit it; the codex sandbox can execute outside its worktree.
RUNSH="$HERE/scripts/run.sh"

case $GROUP in
  vault) OWNED='`quartz/components/scripts/base-vault.inline.ts`, `quartz/util/base/vault/**`, `quartz/styles/pages/vault.scss`, `quartz/styles/pages/vault/**`, `quartz/components/scripts/base-kanban.inline.ts`, `quartz/util/base/kanban*`' ;;
  views) OWNED='`quartz/components/scripts/base-bookcase.inline.ts`, `quartz/util/base/bookcase/**`, `quartz/styles/pages/bookcase.scss`, `quartz/components/scripts/base-query.inline.ts`, `quartz/components/scripts/base-search.inline.ts`, `quartz/components/scripts/base-view-selector.inline.ts`, `quartz/util/base/table*`, `quartz/util/base/cards*`, `quartz/util/base/list*`' ;;
  triathlon) OWNED='`quartz/components/scripts/triathlon.inline.ts`, `quartz/components/triathlon/**` except `maps/`, `quartz/components/styles/triathlon*.scss`, `quartz/components/styles/tri-*.scss`' ;;
  maps) OWNED='`quartz/components/triathlon/maps/**`, `quartz/components/triathlon/activity/comparison-map.ts`, `quartz/components/triathlon/activity/map-controls.ts`, `quartz/components/scripts/base-map.inline.ts`, `quartz/components/scripts/base-map-view.ts`, `quartz/components/scripts/base-map-data.ts`, `quartz/components/scripts/mapbox-client.ts`, `quartz/components/scripts/triathlon-map-heat.ts`, `quartz/components/styles/map*.scss`' ;;
  *) echo "unknown group $GROUP (vault|views|triathlon|maps)" >&2; exit 2 ;;
esac
SCENARIOS=$(cd "$HERE" && python3 -c "import scenarios; print(', '.join(scenarios.GROUPS['$GROUP']))")

URL=$("$HERE/worktree.sh" up "$NAME" "$PORT")
WT=$("$HERE/worktree.sh" path "$NAME")
echo "$GROUP" > "$WT/.base-perf/group"
# The run lives inside the worktree (a sandboxed worker can only write there; .base-perf is
# excluded from git status) and the pass directory links to it.
RUN_DIR="$WT/.base-perf/run-$(basename "$RUN")"
mkdir -p "$RUN_DIR" "$RUN"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
ln -sfn "$RUN_DIR" "$RUN/$NAME"
BRANCH=$(git -C "$WT" branch --show-current)
COMMIT=$(git -C "$WT" rev-parse --short HEAD)
git -C "$WT" status --short > "$RUN_DIR/status-before.txt"

echo "baseline: $NAME ($GROUP) at $URL" >&2
( cd "$WT" && "$HERE/bench.py" --base-url "$URL" --group "$GROUP" --reps 5 --lighthouse 2 --out "$RUN_DIR/baseline" --session "bp-$NAME" ) > "$RUN_DIR/baseline.log" 2>&1
( cd "$WT" && "$HERE/scripts/run.sh" check "$NAME" ) > "$RUN_DIR/baseline/checks.txt" 2>&1 || true

sed -e "s|{{GROUP}}|$GROUP|g" -e "s|{{WORKTREE}}|$WT|g" -e "s|{{BRANCH}}|$BRANCH|g" -e "s|{{COMMIT}}|$COMMIT|g" \
    -e "s|{{WORKER}}|$WORKER|g" -e "s|{{URL}}|$URL|g" -e "s|{{RUN}}|$RUNSH|g" -e "s|{{NAME}}|$NAME|g" \
    -e "s|{{SCENARIOS}}|$SCENARIOS|g" -e "s|{{RUN_DIR}}|$RUN_DIR|g" -e "s|{{MAX_EXPERIMENTS}}|${BP_MAX_EXPERIMENTS:-8}|g" \
    -e "s|{{OWNED}}|$OWNED|g" -e "s|{{HARNESS}}|$HERE|g" "$HERE/task.md" > "$RUN_DIR/task.txt"
: > "$RUN_DIR/experiments.md"
rm -f "$RUN_DIR/exit-code"

echo "worker: $WORKER on $NAME" >&2
rc=0
case $WORKER in
  sol | astra)
    [ "$WORKER" = sol ] && MODEL=gpt-6.1-sol || MODEL=gpt-6-astra
    # workspace-write keeps edits and commands inside the worktree; the task text holds the rest.
    ( cd "$WT" && codex exec -C "$WT" -s workspace-write --skip-git-repo-check -m "$MODEL" \
        -c model_reasoning_effort="${BP_CODEX_EFFORT:-medium}" -c sandbox_workspace_write.network_access=true \
        --json -o "$RUN_DIR/last-message.txt" - \
        < "$RUN_DIR/task.txt" > "$RUN_DIR/events.ndjson" 2> "$RUN_DIR/stderr.txt" ) || rc=$?
    ;;
  fable)
    ENV="$WS/.env"
    KEY=$(rg -m1 '^CLAUDE_API_KEY=' "$ENV" | cut -d= -f2- | tr -d "\"'")
    [ -n "$KEY" ] || { echo "CLAUDE_API_KEY is not in .env" >&2; exit 2; }
    WSID=$(rg -m1 '^CLAUDE_WORKSPACE_ID=' "$ENV" | cut -d= -f2- | tr -d "\"'" || true)
    # --bare authenticates with ANTHROPIC_API_KEY only, so the run bills the API credits and not
    # the subscription. The worktree is the cwd; the tool allowlist keeps it inside the loop.
    ( cd "$WT" && ANTHROPIC_API_KEY=$KEY ANTHROPIC_CUSTOM_HEADERS=${WSID:+"anthropic-workspace-id: $WSID"} \
      claude -p --bare --no-session-persistence --model claude-fable-5-1 --effort "${BP_FABLE_EFFORT:-medium}" \
        --output-format stream-json --verbose --max-budget-usd "${BP_BUDGET_USD:-60}" --permission-mode dontAsk \
        --allowedTools 'Read' 'Edit' 'Write' 'Grep' 'Glob' "Bash($HERE/*)" 'Bash(rg *)' 'Bash(fd *)' \
          'Bash(sed -n *)' 'Bash(cat *)' 'Bash(head *)' 'Bash(tail *)' 'Bash(wc *)' 'Bash(ls *)' 'Bash(git status*)' \
          'Bash(git diff*)' 'Bash(git log *)' 'Bash(git show *)' 'Bash(git checkout -- *)' 'Bash(pnpm exec tsgo *)' \
          'Bash(pnpm exec oxlint *)' 'Bash(pnpm exec oxfmt *)' 'Bash(pnpm test *)' 'Bash(curl *)' \
        --disallowedTools 'Agent' 'WebFetch' 'WebSearch' 'NotebookEdit' \
        < "$RUN_DIR/task.txt" > "$RUN_DIR/events.ndjson" 2> "$RUN_DIR/stderr.txt" ) || rc=$?
    jq -r 'select(.type == "result") | .result // empty' "$RUN_DIR/events.ndjson" > "$RUN_DIR/last-message.txt" 2> /dev/null || true
    ;;
  argon)
    ( cd "$WT" && argon --workspace "$WT" --browser none --stream --timeout "${BP_TIMEOUT:-0}" \
        < "$RUN_DIR/task.txt" > "$RUN_DIR/events.ndjson" 2> "$RUN_DIR/stderr.txt" ) || rc=$?
    python3 "$HOME/.claude/skills/argon/scripts/inspect_run.py" "$RUN_DIR/events.ndjson" --exit-code "$rc" \
      --stderr "$RUN_DIR/stderr.txt" > "$RUN_DIR/inspect.txt" 2>&1 || true
    ;;
  caw)
    ( cd "$WT" && caw --workspace "$WT" --model "${BP_CAW_MODEL:-grok-4.7-medium}" --trust --force --browser none \
        --stream --timeout "${BP_TIMEOUT:-0}" \
        < "$RUN_DIR/task.txt" > "$RUN_DIR/events.ndjson" 2> "$RUN_DIR/stderr.txt" ) || rc=$?
    python3 "$HOME/.claude/skills/caw/scripts/inspect_run.py" "$RUN_DIR/events.ndjson" --exit-code "$rc" \
      --stderr "$RUN_DIR/stderr.txt" > "$RUN_DIR/inspect.txt" 2>&1 || true
    ;;
  *) echo "unknown worker $WORKER (sol|astra|fable|argon|caw)" >&2; exit 2 ;;
esac
echo "$rc" > "$RUN_DIR/exit-code"
git -C "$WT" status --short > "$RUN_DIR/status-after.txt"
git -C "$WT" diff > "$RUN_DIR/changes.patch"
"$HERE/worktree.sh" down "$NAME"
echo "done: $NAME rc=$rc, $(wc -l < "$RUN_DIR/changes.patch") patch lines" >&2
