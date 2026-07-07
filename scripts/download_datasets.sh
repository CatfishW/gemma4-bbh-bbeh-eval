#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/benwulab/gemma4-eval/datasets}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

if [ ! -d BIG-Bench-Hard/.git ]; then
  git clone --depth 1 https://github.com/suzgunmirac/BIG-Bench-Hard.git
else
  git -C BIG-Bench-Hard pull --ff-only
fi

if [ ! -d bbeh/.git ]; then
  git clone --depth 1 https://github.com/google-deepmind/bbeh.git
else
  git -C bbeh pull --ff-only
fi

if [ ! -d unpuzzles_and_simple_reasoning/.git ]; then
  git clone --depth 1 https://github.com/google-deepmind/unpuzzles_and_simple_reasoning.git
else
  git -C unpuzzles_and_simple_reasoning pull --ff-only
fi

echo "BBH  $(git -C BIG-Bench-Hard rev-parse HEAD)"
echo "BBEH $(git -C bbeh rev-parse HEAD)"
echo "USR  $(git -C unpuzzles_and_simple_reasoning rev-parse HEAD)"
