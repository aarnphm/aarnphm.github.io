#!/usr/bin/env bash

NODE_ENV=production pnpm exec quartz/bootstrap-cli.mjs build --concurrency 10 --bundleInfo --verbose

fd --glob "*.pdf" public -x rm
fd --glob "*.ddl" public -x rm

pnpm wrangler deploy
