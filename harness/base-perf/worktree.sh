#!/bin/sh
# A worker's checkout: a `w` worktree with its own pnpm install (store hard links), its
# .quartz-cache cloned (APFS clonefile, so a warm cache costs no space), its own dev watcher on
# its own port, and a log the worker tails for build:ready.
#   harness/base-perf/worktree.sh up <name> <port>     create or reuse, start the watcher, wait for
#                                                      the first build:ready and wrangler Ready
#   harness/base-perf/worktree.sh down <name>          stop the watcher
#   harness/base-perf/worktree.sh wait-build <name>    wait for a build:ready newer than the last
#                                                      wait (or the one in flight), then HTTP 200
#   harness/base-perf/worktree.sh url <name>           the wrangler URL the bench should hit
#   harness/base-perf/worktree.sh path <name>          the worktree path
# The page server is wrangler at port + 707 (quartz/scripts/dev.ts); the watcher's stdout goes to
# <worktree>/.base-perf/dev.log and its pid to <worktree>/.base-perf/dev.pid.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
WS=$(git -C "$HERE" rev-parse --show-toplevel)
cmd=${1:-}; name=${2:-}
[ -n "$cmd" ] && [ -n "$name" ] || { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
WT="$HOME/workspace/worktrees/garden/bp-$name"
STATE="$WT/.base-perf"

port_of() { cat "$STATE/port"; }
url_of() { echo "http://localhost:$(( $(port_of) + 707 ))"; }

case $cmd in
  path) echo "$WT" ;;
  url) url_of ;;
  up)
    port=${3:?port}
    if [ ! -d "$WT" ]; then
      # `w` opens an interactive shell in the new worktree; closed stdin makes it return.
      w garden "bp-$name" < /dev/null > /dev/null 2>&1 || true
      [ -d "$WT" ] || { echo "w did not create $WT" >&2; exit 1; }
    fi
    mkdir -p "$STATE"
    echo "$port" > "$STATE/port"
    # Workers start from the main checkout's HEAD: a worker branch never has commits of its own,
    # so this is a fast-forward; a tree with edits (a run in progress) is left where it is.
    head0=$(git -C "$WT" rev-parse HEAD)
    if [ -z "$(git -C "$WT" status --porcelain)" ]; then
      git -C "$WT" merge --ff-only -q "$(git -C "$WS" rev-parse HEAD)" 2> /dev/null \
        || echo "worktree $WT is not behind main's HEAD on a straight line; left as is" >&2
    fi
    # Worktrees share the main repo's info/exclude; the state dir must stay out of status, diff
    # and the ab stash.
    exclude="$WS/.git/info/exclude"
    rg -q '^\.base-perf/$' "$exclude" 2> /dev/null || echo '.base-perf/' >> "$exclude"
    # A real install from the store (hard links, offline when possible): pnpm's run-time deps check
    # treats a linked node_modules as foreign and would try to purge it.
    [ -x "$WT/node_modules/.bin/tsx" ] || ( cd "$WT" && CI=true pnpm install --frozen-lockfile --prefer-offline > "$STATE/install.log" 2>&1 ) \
      || { tail -n 20 "$STATE/install.log" >&2; exit 1; }
    [ -e "$WT/.env" ] || ln -s "$WS/.env" "$WT/.env"
    # Clone both warm caches: the repo root one (build stages, citation trees) and
    # quartz/.quartz-cache (store payloads: citations.json, strava.json, garmin.json, covers,
    # og-images; without citations.json an arXiv link resolves to a key the bibliography lacks
    # and the build fails). The build salt is the same at the same source, so the first build is
    # incremental instead of cold. `-c` is clonefile, `-n` keeps what the worktree already
    # generated; both fall back to a plain copy on other filesystems.
    for cache in .quartz-cache quartz/.quartz-cache; do
      [ -d "$WS/$cache" ] || continue
      mkdir -p "$WT/$cache"
      cp -Rcn "$WS/$cache/." "$WT/$cache/" 2> /dev/null || cp -Rn "$WS/$cache/." "$WT/$cache/" 2> /dev/null || true
    done
    rm -rf "$WT/.quartz-cache/output-stage-public-"* "$WT/.quartz-cache/output-receipt-public.json"
    # Bibliographies are data other sessions stage before their notes land; HEAD alone may not
    # build. Take the main checkout's copy, never a source file.
    for bib in $(cd "$WS" && git ls-files 'content/*.bib' 'content/**/*.bib'); do
      cmp -s "$WS/$bib" "$WT/$bib" 2> /dev/null || cp "$WS/$bib" "$WT/$bib"
    done
    # A watcher left from an earlier run serves the build of the commit it started on; after a
    # fast-forward the baseline would measure that stale build while a rebuild reloads the pages.
    if [ "$head0" != "$(git -C "$WT" rev-parse HEAD)" ] && [ -f "$STATE/dev.pid" ]; then
      echo "worktree moved to $(git -C "$WT" rev-parse --short HEAD); restarting its watcher" >&2
      "$0" down "$name"
    fi
    if [ -f "$STATE/dev.pid" ] && kill -0 "$(cat "$STATE/dev.pid")" 2> /dev/null; then
      echo "watcher already running (pid $(cat "$STATE/dev.pid"))" >&2
    else
      : > "$STATE/dev.log"
      ( cd "$WT" && exec pnpm exec tsx quartz/scripts/dev.ts --port "$port" --serve ) > "$STATE/dev.log" 2>&1 &
      echo $! > "$STATE/dev.pid"
    fi
    rm -f "$STATE/wait-build.pos"
    "$0" wait-build "$name"
    url_of
    ;;
  down)
    # pnpm → tsx → dev.ts → pnpm → wrangler → workerd: wrangler reparents to launchd when dev.ts
    # dies, keeps the page port and makes the next watcher loop on "port is occupied". Walk the
    # tree from the leaves, then whatever still listens on the page port.
    if [ -f "$STATE/dev.pid" ]; then
      kill_tree() {
        for child in $(pgrep -P "$1" 2> /dev/null); do kill_tree "$child"; done
        kill -TERM "$1" 2> /dev/null || true
      }
      kill_tree "$(cat "$STATE/dev.pid")"
      rm -f "$STATE/dev.pid"
    fi
    sleep 1
    if [ -f "$STATE/port" ]; then
      for pid in $(lsof -ti "tcp:$(( $(port_of) + 707 ))" -sTCP:LISTEN 2> /dev/null); do
        kill -TERM "$pid" 2> /dev/null || true
      done
      pkill -TERM -f "wrangler dev --name portfolio-dev --port $(( $(port_of) + 707 ))" 2> /dev/null || true
    fi
    ;;
  wait-build)
    log="$STATE/dev.log"
    pos="$STATE/wait-build.pos"
    last=$(cat "$pos" 2> /dev/null || echo 0)
    # Builds the worker did not wait for yet count as new; otherwise wait for the next one.
    deadline=$(( $(date +%s) + 900 ))
    while :; do
      count=$(rg -c '"type":"build:ready"' "$log" 2> /dev/null || echo 0)
      if [ "$count" -gt "$last" ]; then
        echo "$count" > "$pos"
        break
      fi
      if rg -q "build:error|Error: |failed with exit" "$log" 2> /dev/null && [ "$count" -eq "$last" ]; then
        tail -n 20 "$log" >&2
      fi
      [ "$(date +%s)" -lt "$deadline" ] || { echo "no build:ready in 900 s; tail of $log:" >&2; tail -n 30 "$log" >&2; exit 1; }
      sleep 2
    done
    url=$(url_of)
    for _ in $(seq 1 120); do
      code=$(curl -s -o /dev/null -w '%{http_code}' "$url/" || echo 000)
      [ "$code" = 200 ] && { echo "build:ready $count, $url up" >&2; exit 0; }
      sleep 2
    done
    echo "$url not serving after build:ready" >&2
    exit 1
    ;;
  *) echo "unknown command: $cmd" >&2; exit 2 ;;
esac
