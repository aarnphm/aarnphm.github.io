#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
if [ "${1:-}" = test ]; then
  shift
  exec uv run --no-project --offline python harness/bookcase-spines/e2e.py "$@"
fi
exec uv run --no-project --offline python harness/bookcase-spines/cli.py "$@"
