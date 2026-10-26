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

run_stage() {
  local stage="$1"
  shift
  local started=$SECONDS
  local status
  printf '__QUARTZ_DEPLOY_STAGE__{"stage":"%s","state":"started","at":"%s"}\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  printf '__QUARTZ_DEPLOY_STAGE__{"stage":"%s","state":"finished","elapsedSeconds":%d,"exitCode":%d}\n' "$stage" "$((SECONDS - started))" "$status"
  return "$status"
}

deploy() {
  load_env
  GIT_LFS_SKIP_SMUDGE=1 run_stage git-pull git pull
  run_stage lfs-install git lfs install --local
  run_stage lfs-pull git lfs pull
  run_stage lfs-required git lfs pull --include="quartz/runtime/native/packs/**,content/triathlon/wind/**/*.json,content/thoughts/images/**,content/hinterland/app/**" --exclude=""
  run_stage lfs-checkout git lfs checkout
  run_stage wind-validation pnpm exec tsx quartz/scripts/validate-wind-archive.ts

  export GITHUB_SHA="$(git rev-parse HEAD)"

  EMAIL_EMITTER_ENABLED=1 NODE_ENV=production run_stage quartz-build pnpm exec quartz/bootstrap-cli.mjs build --concurrency 16 --bundleInfo --verbose

  run_stage prune-ddl fd --glob "*.ddl" public -x rm
  run_stage prune-war fd --glob "*.war" public -x rm
  run_stage prune-embeddings rm public/embeddings-text.jsonl

  run_stage pace-model pnpm model:retrain || echo "pace model refresh failed; deploying site without model update"

  run_stage wrangler pnpm wrangler deploy --minify
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    "" | deploy)
      if [[ "${GARDEN_DEPLOY_LOG_ACTIVE:-}" != 1 ]]; then
        exec node quartz/scripts/deploy-log.mjs -- bash "$0" "$@"
      fi
      deploy
      ;;
    *)
      echo "usage: $0 [deploy]" >&2
      exit 2
      ;;
  esac
fi
