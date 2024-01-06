#!/usr/bin/env bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

if [[ ! -d quartz.git ]]; then
  gh repo clone jackyzha0/quartz -- --bare
fi

git --git-dir=quartz.git --work-tree=. checkout HEAD -- quartz
