#!/usr/bin/env bash
# Run the full test matrix: Python pytest, Rust cargo test, Go vet/build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== Python =="
if [[ ! -d "$ROOT/.venv" ]]; then
    python3 -m venv "$ROOT/.venv"
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT/ai_core/requirements.txt" pytest
( cd "$ROOT" && python -m pytest tests/ -v )
deactivate

echo "== Rust =="
( cd "$ROOT/performance" && cargo test --release )

echo "== Go =="
( cd "$ROOT/api-gateway" && go mod tidy && go vet ./... && go build ./... )

echo "✓ all green"
