#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/jupyter" ]]; then
  echo "Missing .venv/bin/jupyter. Create the virtual environment and install requirements first."
  exit 1
fi

echo "Starting JupyterLab without auto-opening a browser."
echo "Open the printed URL manually in Safari or Firefox."

.venv/bin/jupyter lab \
  --no-browser \
  --ServerApp.open_browser=False \
  --ServerApp.port=8888
