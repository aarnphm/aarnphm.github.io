#!/usr/bin/env bash
set -euo pipefail

printf '\n==> Hydrating Git LFS blobs\n'
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs binary is missing" >&2
  exit 1
fi

git lfs install --local

branch="${CF_PAGES_BRANCH:-main}"
branch="${branch#[}"
branch="${branch%]}"
ref="${CF_PAGES_COMMIT_SHA:-}"

if [[ -n "$ref" ]]; then
  if ! git lfs fetch origin "$ref"; then
    echo "git lfs fetch by commit failed, falling back to branch '${branch}'"
    git lfs fetch origin "$branch"
  fi
else
  git lfs fetch origin "$branch"
fi

# As a safety net, ensure we have all tracked blobs for this tree
if ! git lfs checkout; then
  echo "git lfs checkout failed, attempting a full pull"
  git lfs pull origin "$branch"
fi

git lfs ls-files --size

printf '\n==> Installing workspace dependencies\n'
pnpm install --frozen-lockfile

printf '\n==> Building Quartz bundle for Pages\n'
CF_PAGES=1 pnpm prod

printf '\n==> Embedding shard size\n'
stat --format='%n %s bytes' content/embeddings/vectors-000.bin
