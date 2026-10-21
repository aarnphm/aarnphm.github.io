#!/bin/sh
# Stage 2: one read-only boundary proposal per target from one worker, in parallel.
#   harness/token-budget/dispatch.sh <run-dir> <sol|opus|argon> [<slug>...]
# Each <run-dir>/<slug>/brief.md becomes <slug>/<worker>/task.txt; the run writes events.ndjson,
# stderr.txt and exit-code there (Codex and Claude also write last-message.txt). Workers read the tree at
# TB_WORKSPACE (default: this checkout), so a `w` worktree can be the source as well.
# TB_ARGON_TIMEOUT sets Argon's wall-clock limit in seconds (7200; 0 waits until completion).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
WS=${TB_WORKSPACE:-$(git -C "$HERE" rev-parse --show-toplevel)}
[ $# -ge 2 ] || { echo "usage: $0 <run-dir> <sol|opus|argon> [<slug>...]" >&2; exit 2; }
RUN=$(cd "$1" && pwd)
W=$2
shift 2

case $W in
  sol)
    ID='codex:gpt-6.1-sol'
    CONSTRAINT='minimise-interface'
    TEXT='Minimise the interface: aim for the fewest names the production importers need (1 to 3 where possible) and the most leverage per name. Hide every export that production callers do not use, and plan to move tests that import internals to the entry interface.'
    ;;
  opus)
    # Takes Sol's constraint, so `TB_WORKERS="opus argon"` replaces Sol one for one.
    ID="claude:claude-opus-5-5-${TB_OPUS_EFFORT:-high}"
    CONSTRAINT='minimise-interface'
    TEXT='Minimise the interface: aim for the fewest names the production importers need (1 to 3 where possible) and the most leverage per name. Hide every export that production callers do not use, and plan to move tests that import internals to the entry interface.'
    # --bare authenticates with ANTHROPIC_API_KEY only (never OAuth), so the run bills the API
    # credits in the main checkout's .env and not the subscription quota.
    ENV="$(git -C "$HERE" rev-parse --show-toplevel)/.env"
    KEY=$(rg -m1 '^CLAUDE_API_KEY=' "$ENV" | cut -d= -f2- | tr -d "\"'")
    [ -n "$KEY" ] || { echo "CLAUDE_API_KEY is not in .env" >&2; exit 2; }
    # A key that is not scoped to a workspace (sk-ant-usr-...) must name one in each request.
    WSID=$(rg -m1 '^CLAUDE_WORKSPACE_ID=' "$ENV" | cut -d= -f2- | tr -d "\"'" || true)
    ;;
  argon)
    TB_ARGON_TIMEOUT=${TB_ARGON_TIMEOUT:-7200}
    case $TB_ARGON_TIMEOUT in
      *[!0-9]*) echo "TB_ARGON_TIMEOUT must be a nonnegative integer in seconds (0 means unlimited)" >&2; exit 2 ;;
    esac
    ID='argon:gemini-4-argon-high'
    CONSTRAINT='common-caller'
    TEXT='Optimise for the most common caller: the production importers listed below should keep a trivial default call, even if the interface keeps a few more names. Group the implementation so that the change you expect most often touches one file.'
    ;;
  *) echo "unknown worker: $W" >&2; exit 2 ;;
esac

[ $# -gt 0 ] || set -- $(cd "$RUN" && for b in */brief.md; do dirname "$b"; done)

for slug in "$@"; do
  d="$RUN/$slug/$W"
  mkdir -p "$d"
  {
    cat <<EOF
# Assignment: module boundary proposal (token budget refactor, stage 2)

Workspace: $WS. You are read-only: do not create, edit, stage, move or delete any file. Other agents are working in this tree; leave everything as you found it.
Worker id: \`$ID\`. Constraint id: \`$CONSTRAINT\`.
Design constraint: $TEXT
Allowed commands: read files (\`sed -n\`, \`cat\`), search with \`rg\` and \`fd\`, structural queries with \`ast-grep run -p '<pattern>' -l ts\` (no \`--rewrite\`, \`-U\` or \`scan --update-all\`), and \`git log\`/\`git show\` on the target and its importers. No builds, tests, installs, browsers or sub-agents.
Deliverable: your final message ends with exactly one \`\`\`json fenced block holding the proposal (schema under "Output" below; "the path given in your assignment" means this final message, not a file). Before the block, at most 6 lines: outcome (\`passed\`, \`failed\` or \`incomplete\`), the main trade-off, and anything you could not decide. Nothing after the block.

EOF
    cat "$RUN/$slug/brief.md"
  } > "$d/task.txt"
  rm -f "$d/exit-code" "$d/last-message.txt"
  case $W in
    sol)
      ( cd "$d" && codex exec -C "$WS" -s read-only -m gpt-6.1-sol -c model_reasoning_effort=high \
          --json -o last-message.txt - < task.txt > events.ndjson 2> stderr.txt; echo $? > exit-code ) &
      ;;
    opus)
      ( cd "$WS" || exit 1
        rc=0
        ANTHROPIC_API_KEY=$KEY ANTHROPIC_CUSTOM_HEADERS=${WSID:+"anthropic-workspace-id: $WSID"} \
          claude -p --bare --no-session-persistence \
          --model claude-opus-5-5 --effort "${TB_OPUS_EFFORT:-high}" --output-format json \
          --permission-mode dontAsk \
          --allowedTools 'Read' 'Grep' 'Glob' 'Bash(rg *)' 'Bash(fd *)' 'Bash(ast-grep run *)' \
            'Bash(git log *)' 'Bash(git show *)' 'Bash(sed -n *)' 'Bash(cat *)' 'Bash(wc *)' \
          --disallowedTools 'Edit' 'Write' 'NotebookEdit' 'Agent' 'WebFetch' 'WebSearch' \
          < "$d/task.txt" > "$d/events.json" 2> "$d/stderr.txt" || rc=$?
        # The json output is the event list; the last event is the result.
        jq -r 'if type == "array" then .[-1] else . end | select(.is_error | not) | .result // empty' \
          "$d/events.json" > "$d/last-message.txt" 2> /dev/null || true
        echo $rc > "$d/exit-code" ) &
      ;;
    argon)
      # agy can keep running after its result event; stop this run (and only it) after a grace period.
      ( cd "$d"
        printf '%s\n' "$TB_ARGON_TIMEOUT" > timeout-seconds
        argon --workspace "$WS" --browser none --stream --timeout "$TB_ARGON_TIMEOUT" \
          < task.txt > events.ndjson 2> stderr.txt &
        pid=$!
        while kill -0 "$pid" 2> /dev/null; do
          if rg -q '"event":\s*"result"' events.ndjson 2> /dev/null; then
            sleep 30
            kill -0 "$pid" 2> /dev/null && { pkill -TERM -P "$pid"; kill -TERM "$pid"; echo "killed after result" > watchdog; }
            break
          fi
          sleep 10
        done
        # set -e would end this subshell when wait reports the watchdog's SIGTERM (143).
        rc=0; wait "$pid" || rc=$?; echo $rc > exit-code ) &
      ;;
  esac
done
wait
