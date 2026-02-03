#!/usr/bin/env bash

set -e

git pull
git lfs pull && git lfs checkout

export GITHUB_SHA="$(git rev-parse HEAD)"

NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 10 --bundleInfo --verbose

fd --glob "*.ddl" public -x rm
fd --glob "*.war" public -x rm
rm public/embeddings-text.jsonl

pnpm wrangler deploy --minify

unset GITHUB_SHA
