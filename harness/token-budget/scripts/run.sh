#!/bin/sh
# One entry point for the token budget harness (.agents/skills/token-budget/SKILL.md).
#   harness/token-budget/scripts/run.sh <command> [args]
#
#   check [--all | --paths <p>...] [--json]    the pre-commit hook (o200k_base; tests exempt)
#   pass [--top N] [--run <dir>] [<path>...]   stages 0-4: inventory, Sol and Argon proposals,
#                                              validation, agent-decide (TB_WORKERS= baseline only,
#                                              TB_WORKERS="opus argon" Opus on API credits for Sol)
#   summary <run>                              per target: proposals, checks, decisions, ARI
#   worktree <name> <path>                     w garden tb-<name>; checks the target blob at HEAD
#   move <worktree> <run>/<slug> <worker>      stage 5: the worker's proposal, blocks moved verbatim,
#                                              import headers and entry re-exports generated
#   proof <worktree> <path>                    move proof: code lines, original vs module
#   css <worktree> <path>                      compiled CSS, HEAD vs worktree, for the target and
#                                              each stylesheet that @uses it; must be identical
#   land <worktree> <out-dir> <path>...        worktree changes vs HEAD -> <out-dir>/land.patch,
#                                              applied to this checkout's index (stage only)
#   clean <worktree>                           after the commit: remove the worktree and its branch
#                                              when every changed file matches HEAD
#   inventory | dispatch | collect | validate | baseline [args]   one stage script, as is
#
# Commands run from the main checkout; <worktree> is a `w` worktree path.
set -eu
TARGET=$0
while [ -L "$TARGET" ]; do
  DIR=$(cd -P "$(dirname "$TARGET")" && pwd)
  TARGET=$(readlink "$TARGET")
  [ "${TARGET#/}" = "$TARGET" ] && TARGET="$DIR/$TARGET"
done
SCRIPT_DIR=$(cd -P "$(dirname "$TARGET")" && pwd)
TB=$(cd "$SCRIPT_DIR/.." && pwd)
WS=$(git -C "$TB" rev-parse --show-toplevel)
SASS="$WS/node_modules/.bin/sass"

usage() { sed -n '3,22p' "$TARGET" | sed 's/^# \{0,1\}//' >&2; exit 2; }
need() { [ "$1" -ge "$2" ] || usage; }

cmd=${1:-}
[ -n "$cmd" ] || usage
shift

case $cmd in
  check) exec "$TB/check.py" "$@" ;;
  pass) exec "$TB/pass.sh" "$@" ;;
  inventory | validate | baseline) exec "$TB/$cmd.py" "$@" ;;
  dispatch | collect) exec "$TB/$cmd.sh" "$@" ;;

  summary)
    need $# 1
    for v in "$1"/*/validation.json; do
      [ -f "$v" ] || continue
      s=$(dirname "$v")
      echo
      jq -r '.target' "$v"
      jq -r '.results[] | [.worker, .status, ([.checks[] | select(.status == "fail") | .id] | join(",")), (.metrics | "files=\(.files) largest=\(.largest) exports \(.exportsBefore)->\(.exportsAfter) cutShare=\(.cutShare) sharedState=\(.sharedState)")] | join("|")' "$v" \
        | while IFS='|' read -r worker status fails metrics; do
            d="$s/decisions/$(printf '%s' "$worker" | sed 's/[^A-Za-z0-9_.-][^A-Za-z0-9_.-]*/-/g').decision.json"
            action=$(jq -r '.action // "?"' "$d" 2> /dev/null || echo '?')
            printf '  %-28s %-5s decide=%-9s %s %s\n' "$worker" "$status" "$action" "$metrics" "${fails:+fail: $fails}"
          done
      jq -r '.agreement[] | "  ARI \(.a) vs \(.b): \(.ari)"' "$v"
    done
    echo
    ;;

  worktree)
    need $# 2
    wt="$HOME/workspace/worktrees/garden/tb-$1"
    # `w` opens an interactive shell in the new worktree; closed stdin makes it return.
    [ -d "$wt" ] || w garden "tb-$1" < /dev/null > /dev/null 2>&1
    [ -d "$wt" ] || { echo "w did not create $wt" >&2; exit 1; }
    head=$(git -C "$wt" rev-parse "HEAD:$2")
    index=$(git -C "$WS" rev-parse ":$2")
    echo "$wt (branch $(git -C "$wt" branch --show-current))"
    [ "$head" = "$index" ] \
      || echo "warning: $2 in this checkout's index ($index) differs from HEAD ($head); wait for a commit" >&2
    ;;

  proof) need $# 2; exec "$TB/proof.py" "$@" ;;
  move)
    need $# 3
    exec "$TB/move.py" "$(dirname "$2")/graph.json" "$2/$3/proposal.json" "$1"
    ;;

  css)
    need $# 2
    wt=$1 p=$2
    tmp=$(mktemp -d)
    # HEAD's stylesheets in a scratch tree, so relative @use paths resolve as they did.
    git -C "$wt" archive HEAD -- $(git -C "$wt" ls-tree -r --name-only HEAD -- quartz | rg '\.s[ac]ss$') | tar -x -C "$tmp"
    base=$(basename "${p%.*}")
    users=$(cd "$wt" && rg -l "@use ['\"][^'\"]*/_?$base(\.scss)?['\"]" -g '*.scss' quartz | rg -v "^${p%.*}/" || true)
    status=0
    for f in "$p" $users; do
      for style in expanded compressed; do
        if "$SASS" --no-source-map --style=$style "$tmp/$f" > "$tmp/a.css" \
          && "$SASS" --no-source-map --style=$style "$wt/$f" > "$tmp/b.css" \
          && cmp -s "$tmp/a.css" "$tmp/b.css"; then
          echo "identical  $style  $f ($(wc -c < "$tmp/b.css" | tr -d ' ') bytes)"
        else
          echo "DIFFERENT  $style  $f"; status=1
        fi
      done
    done
    rm -r "$tmp"
    exit $status
    ;;

  land)
    need $# 3
    wt=$1 out=$2
    shift 2
    mkdir -p "$out"
    git -C "$WS" diff --cached --quiet HEAD -- "$@" \
      || { echo "these paths are already staged here; land the increment by hand" >&2; exit 1; }
    # `git add <dir>` skips ignored files silently (image-sharing.ts matched `image-*`); rename them.
    ignored=$(git -C "$wt" ls-files --others --ignored --exclude-standard -- "$@")
    if [ -n "$ignored" ]; then
      echo "these module files are git-ignored and would not land; rename them:" >&2
      printf '%s\n' "$ignored" | git -C "$wt" check-ignore -v --stdin >&2
      exit 1
    fi
    git -C "$wt" add -- "$@"
    git -C "$wt" diff --cached --binary HEAD -- "$@" > "$out/land.patch"
    git -C "$WS" apply --index --3way "$out/land.patch"
    git -C "$WS" status --short -- "$@"
    echo "staged from $out/land.patch; after aarnphm commits: run.sh clean $wt"
    ;;

  clean)
    need $# 1
    wt=$1
    # The split lives in the worktree as uncommitted files (land copies it by patch), so `w --rm`
    # refuses; remove only when every changed file matches this checkout's HEAD.
    left=0
    for f in $(git -C "$wt" status --porcelain -uall | awk '{print $NF}' | rg -v '^node_modules$'); do
      a=$( [ -f "$wt/$f" ] && git -C "$wt" hash-object "$wt/$f" || echo deleted)
      b=$(git -C "$WS" rev-parse -q --verify "HEAD:$f" || echo deleted)
      [ "$a" = "$b" ] || { echo "not in HEAD: $f"; left=1; }
    done
    [ $left -eq 0 ] || { echo "keeping $wt; compare with git diff --no-index, then remove by hand" >&2; exit 1; }
    branch=$(git -C "$wt" branch --show-current)
    rm -f "$wt/node_modules"
    git -C "$WS" worktree remove --force "$wt"
    [ -z "$branch" ] || git -C "$WS" branch -D "$branch"
    ;;

  *) usage ;;
esac
