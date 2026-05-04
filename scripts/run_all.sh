#!/usr/bin/env bash
# Boot the full hybrid AI stack: Rust perf service, Python AI core, Go gateway.
#
# Usage:
#     ./scripts/run_all.sh        # foreground, all three services
#     STOP=1 ./scripts/run_all.sh # stop services started previously
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIDS_FILE="$ROOT/.run_all.pids"

stop_services() {
    if [[ -f "$PIDS_FILE" ]]; then
        echo "→ stopping previously launched services"
        while IFS= read -r pid; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi
}

if [[ "${STOP:-0}" == "1" ]]; then
    stop_services
    exit 0
fi

stop_services
: > "$PIDS_FILE"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ROOT/.env"
    set +a
else
    echo "⚠  .env not found — copy .env.example to .env and add NVIDIA_NIM_API_KEY"
fi

mkdir -p "$ROOT/logs"

# 1. Rust performance service (optional but recommended)
if command -v cargo >/dev/null 2>&1; then
    echo "→ starting Rust perf-server"
    ( cd "$ROOT/performance" && cargo run --release --quiet --bin perf-server ) \
        >"$ROOT/logs/perf.log" 2>&1 &
    echo $! >> "$PIDS_FILE"
else
    echo "⚠  cargo not installed — skipping Rust perf service"
fi

# 2. Python AI core
echo "→ starting Python AI core"
if [[ ! -d "$ROOT/.venv" ]]; then
    python3 -m venv "$ROOT/.venv"
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT/ai_core/requirements.txt"
( cd "$ROOT" && python -m ai_core ) >"$ROOT/logs/ai_core.log" 2>&1 &
echo $! >> "$PIDS_FILE"
deactivate

# 3. Go API gateway
if command -v go >/dev/null 2>&1; then
    echo "→ starting Go API gateway"
    ( cd "$ROOT/api-gateway" && go run . ) \
        >"$ROOT/logs/gateway.log" 2>&1 &
    echo $! >> "$PIDS_FILE"
else
    echo "⚠  go not installed — skipping API gateway"
fi

trap stop_services EXIT INT TERM

echo "✓ stack started. logs: $ROOT/logs/"
echo "  - AI core   → http://localhost:${AI_CORE_PORT:-8000}"
echo "  - Gateway   → http://localhost:${GATEWAY_PORT:-9000}"
echo "  - Perf      → http://localhost:7070"
echo "Press Ctrl+C to stop."
wait
