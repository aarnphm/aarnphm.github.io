#!/usr/bin/env bash

CF_PAGES=1 NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 4 --bundleInfo --verbose

fd --glob "*.pdf" public -x rm
fd --glob "*.ddl" public -x rm

pnpm wrangler deploy
