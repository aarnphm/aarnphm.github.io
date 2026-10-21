#!/bin/sh
# One entry point for the view performance hill climb (harness/base-perf/README.md).
#   harness/base-perf/scripts/run.sh <command> [args]
#
#   bench <name> <out-dir> [--reps N] [--scenario a,b]   the worker's group against its watcher
#   compare <baseline-dir> <candidate-dir>               ACCEPT/REJECT per scenario (exit 3 on reject)
#   wait-build <name>                                    block until the worktree serves a new build
#   check <name>                                         tsgo, oxlint + oxfmt on changed files, view tests
#   ab <name> <out-dir>                                  candidate, then stash + baseline, then restore;
#                                                        writes <out-dir>/{candidate,baseline}/ + compare
#   climb <run-dir> <name> <group> <port> <worker>       one worker end to end (climb.sh)
#   pass [--run <dir>] [--workers v,w,t,m]              the four climbs; workers default to
#                                                        sol,astra,sol,astra (vault, views, triathlon, maps)
#   land <run-dir>/<name>                                apply changes.patch to this checkout's index
#   probe <url> [...]                                    probe.py
# Names are worktrees (harness/base-perf/worktree.sh); the bench always runs in the real Helium
# profile through the CUA broker, one task tab per run.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
WS=$(git -C "$HERE" rev-parse --show-toplevel)
cmd=${1:-}
[ -n "$cmd" ] || { sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
shift
group_of() { cat "$("$HERE/worktree.sh" path "$1")/.base-perf/group" 2> /dev/null || echo "$1"; }

case $cmd in
  bench)
    name=$1; out=$2; shift 2
    wt=$("$HERE/worktree.sh" path "$name")
    [ -d "$wt" ] || wt=$WS
    url=$("$HERE/worktree.sh" url "$name" 2> /dev/null || echo http://localhost:8080)
    ( cd "$wt" && exec "$HERE/bench.py" --base-url "$url" --group "$(group_of "$name")" --out "$out" --session "bp-$name" "$@" )
    ;;
  compare) exec "$HERE/compare.py" "$@" ;;
  wait-build) exec "$HERE/worktree.sh" wait-build "$1" ;;
  probe) exec "$HERE/probe.py" "$@" ;;
  climb) exec "$HERE/climb.sh" "$@" ;;
  check)
    wt=$("$HERE/worktree.sh" path "$1")
    [ -d "$wt" ] || wt=$WS
    cd "$wt"
    changed=$(git diff --name-only; git ls-files --others --exclude-standard)
    rc=0
    pnpm exec tsgo --noEmit || rc=1
    ts=$(printf '%s\n' "$changed" | rg '\.(ts|tsx|mjs)$' || true)
    [ -z "$ts" ] || { pnpm exec oxlint $ts || rc=1; pnpm exec oxfmt --check $ts || rc=1; }
    # The view's own tests; a baseline failure is recorded by climb.sh so the worker may not add one.
    pnpm test quartz/plugins/emitters/base.test.ts || rc=1
    case $(group_of "$1") in
      triathlon | maps) pnpm test quartz/components/triathlon/maps/model.test.ts quartz/components/triathlon/activity/analysis.test.ts || rc=1 ;;
    esac
    exit $rc
    ;;
  ab)
    name=$1; out=$2
    wt=$("$HERE/worktree.sh" path "$name")
    "$0" bench "$name" "$out/candidate" --reps 5
    ( cd "$wt" && git stash push --include-untracked -q -m "base-perf ab $name" )
    "$HERE/worktree.sh" wait-build "$name"
    "$0" bench "$name" "$out/baseline" --reps 5 || true
    ( cd "$wt" && git stash pop -q )
    "$HERE/worktree.sh" wait-build "$name"
    "$0" compare "$out/baseline" "$out/candidate"
    ;;
  pass)
    run=; workers=${BP_WORKERS:-sol,astra,sol,astra}
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; --workers) workers=$2; shift 2 ;; *) break ;; esac; done
    run=${run:-$WS/work/base-perf/pass-$(date +%Y-%m-%d-%H%M)}
    mkdir -p "$run"
    echo "run: $run (workers $workers)" >&2
    # Codex (Sol, Astra) by default; fable bills the API credits, argon and caw their subscriptions.
    IFS=, read -r wv ww wt wm <<EOF2
$workers
EOF2
    "$HERE/climb.sh" "$run" vault vault 7400 "${wv:-sol}" &
    "$HERE/climb.sh" "$run" views views 7500 "${ww:-astra}" &
    "$HERE/climb.sh" "$run" triathlon triathlon 7600 "${wt:-sol}" &
    "$HERE/climb.sh" "$run" maps maps 7700 "${wm:-astra}" &
    wait
    for d in "$run"/*/; do
      n=$(basename "$d")
      printf '%-10s rc=%s patch=%s lines\n' "$n" "$(cat "$d/exit-code" 2> /dev/null || echo ?)" "$(wc -l < "$d/changes.patch" 2> /dev/null || echo 0)"
    done
    ;;
  land)
    d=$1
    [ -s "$d/changes.patch" ] || { echo "no changes in $d" >&2; exit 1; }
    cd "$WS"
    git apply --index --3way "$d/changes.patch" && git status --short
    ;;
  *) echo "unknown command: $cmd" >&2; exit 2 ;;
esac
