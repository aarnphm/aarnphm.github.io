!#/usr/bin/env bash

curl -LsSf https://astral.sh/uv/install.sh | sh

if [[ $VERCEL_ENV == "production" ]]; then
  npx quartz build
else
  npx quartz build --bundleInfo
fi
