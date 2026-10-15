#!/usr/bin/env bash

set -euo pipefail

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
  load_env
  GIT_LFS_SKIP_SMUDGE=1 git pull
  git lfs install --local
  git lfs pull
  git lfs pull --include="quartz/runtime/native/packs/**,content/triathlon/wind/**/*.json" --exclude=""
  git lfs checkout
  pnpm exec tsx quartz/scripts/validate-wind-archive.ts

  export GITHUB_SHA="$(git rev-parse HEAD)"

  EMAIL_EMITTER_ENABLED=1 NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 16 --bundleInfo --verbose

  fd --glob "*.ddl" public -x rm
  fd --glob "*.war" public -x rm
  rm public/embeddings-text.jsonl

  pnpm model:retrain || echo "pace model refresh failed; deploying site without model update"

  pnpm wrangler deploy --minify
}

case "${1:-}" in
  "" | deploy) deploy ;;
  *)
    echo "usage: $0 [deploy]" >&2
    exit 2
    ;;
esac
