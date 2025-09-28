#!/usr/bin/env bash
set -euo pipefail

printf '\n==> Hydrating Git LFS blobs\n'
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs binary is missing" >&2
  exit 1
fi

git lfs install --local
git lfs fetch origin "${CF_PAGES_BRANCH:-main}"
git lfs checkout

git lfs ls-files --size

printf '\n==> Installing workspace dependencies\n'
pnpm install --frozen-lockfile

printf '\n==> Building Quartz bundle for Pages\n'
CF_PAGES=1 pnpm prod

printf '\n==> Embedding shard size\n'
stat --format='%n %s bytes' content/embeddings/vectors-000.bin
