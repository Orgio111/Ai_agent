#!/usr/bin/env bash
# JARVIS Autonomous AI OS — Full Stack Launcher
# Usage:
#   ./scripts/run_all.sh          # Start all services via Docker Compose
#   STOP=1 ./scripts/run_all.sh   # Stop all services
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

stop_services() {
    echo "Stopping JARVIS services..."
    cd "$ROOT/deploy" && docker compose down
}

if [[ "${STOP:-0}" == "1" ]]; then
    stop_services
    exit 0
fi

# Load .env
if [[ -f "$ROOT/.env" ]]; then
    set -a && source "$ROOT/.env" && set +a
else
    echo "WARNING: .env not found — copy .env.example to .env and set NVIDIA_NIM_API_KEY"
fi

# Validate required API key
if [[ -z "${NVIDIA_NIM_API_KEY:-}" ]] || [[ "$NVIDIA_NIM_API_KEY" == nvapi-your* ]]; then
    echo "ERROR: NVIDIA_NIM_API_KEY is not set or is a placeholder." >&2
    echo "       Export a valid key: export NVIDIA_NIM_API_KEY=nvapi-..." >&2
    exit 1
fi

echo "Starting JARVIS infrastructure..."
cd "$ROOT/deploy"
docker compose up -d redis postgres prometheus grafana otel-collector

echo "Waiting for infrastructure to be ready..."
sleep 5

echo "Starting JARVIS microservices..."
docker compose up -d broker memory llm-engine tool-system voice agent-core

echo ""
echo "JARVIS Autonomous AI OS is running:"
echo "  Agent Core  → http://localhost:8000/docs"
echo "  Broker      → http://localhost:8001/health (WS: ws://localhost:8001/ws)"
echo "  LLM Engine  → http://localhost:8002/docs"
echo "  Memory      → http://localhost:8003/docs"
echo "  Tool System → http://localhost:8004/docs"
echo "  Voice       → http://localhost:8005/docs"
echo "  Prometheus  → http://localhost:9090"
echo "  Grafana     → http://localhost:3001 (admin/jarvis)"
echo ""
echo "Start UI:    cd services/ui && npm install && npm run dev"
echo "View logs:   cd deploy && docker compose logs -f --tail=50"
echo "Stop all:    STOP=1 ./scripts/run_all.sh"
