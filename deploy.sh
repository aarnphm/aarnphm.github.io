#!/usr/bin/env bash

NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 10 --bundleInfo --verbose

fd --glob "*.ddl" public -x rm
rm public/embeddings-text.jsonl

pnpm wrangler deploy --minify
