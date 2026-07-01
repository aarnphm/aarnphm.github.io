#!/usr/bin/env bash

set -e

git pull
git lfs install --local
git lfs pull
git lfs pull --include="quartz/runtime/native/packs/**"
git lfs checkout

export GITHUB_SHA="$(git rev-parse HEAD)"

pnpm health:all || exit 1

EMAIL_EMITTER_ENABLED=1 NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 16 --bundleInfo --verbose

fd --glob "*.ddl" public -x rm
fd --glob "*.war" public -x rm
rm public/embeddings-text.jsonl

pnpm model:retrain || echo "pace model refresh failed; deploying site without model update"

pnpm wrangler deploy --minify

unset GITHUB_SHA
