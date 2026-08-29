#!/usr/bin/env bash

set -euo pipefail

readonly HEALTH_STATE="$(git rev-parse --git-path deploy-health-ready)"

health() {
  rm -f "$HEALTH_STATE"
  pnpm wahoo:auth
  pnpm health:all
  git rev-parse HEAD > "$HEALTH_STATE"
}

require_health() {
  if [[ ! -f "$HEALTH_STATE" ]]; then
    echo "deploy blocked: run '$0 health' first" >&2
    exit 1
  fi

  local health_revision
  local current_revision
  health_revision="$(<"$HEALTH_STATE")"
  current_revision="$(git rev-parse HEAD)"
  if [[ "$health_revision" != "$current_revision" ]]; then
    echo "deploy blocked: repository changed since health ran; run '$0 health' again" >&2
    exit 1
  fi
}

load_env() {
  if [[ ! -f .env ]]; then
    echo "deploy blocked: .env is missing" >&2
    exit 1
  fi

  set -a
  source ./.env
  set +a
}

deploy() {
  require_health
  load_env
  git pull
  require_health
  git lfs install --local
  git lfs pull
  git lfs pull --include="quartz/runtime/native/packs/**"
  git lfs checkout

  export GITHUB_SHA="$(git rev-parse HEAD)"

  EMAIL_EMITTER_ENABLED=1 NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 16 --bundleInfo --verbose

  fd --glob "*.ddl" public -x rm
  fd --glob "*.war" public -x rm
  rm public/embeddings-text.jsonl

  pnpm model:retrain || echo "pace model refresh failed; deploying site without model update"

  pnpm wrangler deploy --minify
  rm -f "$HEALTH_STATE"
}

case "${1:-}" in
  health) health ;;
  "" | deploy) deploy ;;
  *)
    echo "usage: $0 [health|deploy]" >&2
    exit 2
    ;;
esac
