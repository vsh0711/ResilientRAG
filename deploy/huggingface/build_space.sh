#!/usr/bin/env bash
# Assemble a directory you can push to a Hugging Face Space.
#   ./deploy/huggingface/build_space.sh
#   cd dist/space && git init -b main && git remote add space https://huggingface.co/spaces/<you>/<name>
#   git add . && git commit -m "deploy" && git push space main --force
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/dist/space"
rm -rf "$out" && mkdir -p "$out"
cp "$root/deploy/huggingface/Dockerfile" "$out/Dockerfile"
cp "$root/deploy/huggingface/SPACE_README.md" "$out/README.md"
cp "$root/backend/pyproject.toml" "$out/"
cp -R "$root/backend/app" "$out/app"
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "Space bundle ready in $out"
