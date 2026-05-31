#!/usr/bin/env bash

set -e

git pull
git lfs install --local
git lfs pull
git lfs pull --include="quartz/runtime/native/packs/**"
git lfs checkout

export GITHUB_SHA="$(git rev-parse HEAD)"

pnpm clean --lockfile && pnpm i && pnpm strava:sync

EMAIL_EMITTER_ENABLED=1 NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 10 --bundleInfo --verbose

fd --glob "*.ddl" public -x rm
fd --glob "*.war" public -x rm
rm public/embeddings-text.jsonl

pnpm wrangler deploy --minify

unset GITHUB_SHA
