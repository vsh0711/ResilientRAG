#!/usr/bin/env bash
# Assemble a directory you can push to a Hugging Face Space.
#   ./deploy/huggingface/build_space.sh            # Gradio SDK, no Docker (default)
#   ./deploy/huggingface/build_space.sh docker     # Docker SDK
#   cd dist/space && git init -b main && git remote add space https://huggingface.co/spaces/<you>/<name>
#   git add . && git commit -m "deploy" && git push space main --force
set -euo pipefail
mode="${1:-gradio}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/dist/space"
rm -rf "$out" && mkdir -p "$out"
cp -R "$root/backend/app" "$out/app"
cp "$root/backend/pyproject.toml" "$out/"
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +

if [ "$mode" = "docker" ]; then
  cp "$root/deploy/huggingface/Dockerfile" "$out/Dockerfile"
  cp "$root/deploy/huggingface/SPACE_README.md" "$out/README.md"
else
  cp "$root/deploy/huggingface/space_app.py" "$out/space_app.py"
  cp "$root/deploy/huggingface/SPACE_README_GRADIO.md" "$out/README.md"
  # requirements.txt from pyproject's dependency list, so the two cannot drift
  awk '/^dependencies = \[/{f=1;next} f&&/^\]/{f=0} f{gsub(/^[ \t]*"|",?[ \t]*$/,"");print}' \
    "$root/backend/pyproject.toml" > "$out/requirements.txt"
  # ZeroGPU's startup check needs the Space's own helper package
  echo "spaces" >> "$out/requirements.txt"
  # the app is imported from this folder, not installed
  rm "$out/pyproject.toml"
fi
echo "Space bundle ($mode) ready in $out"
