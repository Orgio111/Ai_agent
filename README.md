# Hybrid AI System

A JARVIS-class autonomous AI operating system. Multi-agent core in **Python**,
ultra-fast API gateway in **Go**, performance-critical primitives in **Rust**,
all wired to **NVIDIA NIM** as the primary LLM provider with intelligent
model routing.

> ⚡ **Production-ready · multi-language · self-improving**

---

## Architecture

```
                ┌─────────────────────────────┐
                │   Go API Gateway (Fiber)    │   :9000
                │  rate limit · cors · logs   │
                └──────────────┬──────────────┘
                               │ HTTP / stream
                ┌──────────────▼──────────────┐
                │ Python AI Core (FastAPI)    │   :8000
                │  ┌───────────────────────┐  │
                │  │ Orchestrator (loop)   │  │
                │  │  Planner ─► Executor  │  │
                │  │            │          │  │
                │  │      Coder ┘ ─► Critic│  │
                │  └────┬───────────┬──────┘  │
                │       │           │         │
                │  ┌────▼───┐  ┌────▼───┐    │
                │  │  Tools │  │ Memory │    │
                │  │ shell  │  │ short  │    │
                │  │ fs/http│  │ FAISS  │    │
                │  └────┬───┘  └────────┘    │
                └───────┼─────────────────────┘
                        │ HTTP
                ┌───────▼──────────┐    ┌─────────────┐
                │ Rust perf-server │    │ NVIDIA NIM  │
                │ cosine · CRC32   │    │  llama-3    │
                │  parallel rayon  │    │  mistral    │
                └──────────────────┘    │  mixtral    │
                                        └─────────────┘
```

---

## Features

- **NVIDIA NIM as primary LLM** with OpenAI-compatible API.
- **Dynamic model routing** — `complex / balanced / fast / code` tiers chosen
  by heuristics or explicit hint.
- **4-agent loop**: Planner → Executor → Coder → Critic; iterates until the
  Critic accepts (score ≥ 0.75) or `MAX_ITERATIONS` is hit.
- **Memory**:
  - Short-term FIFO buffer per session (50 msgs default).
  - Long-term FAISS vector store with NIM embeddings, persisted on disk.
- **Tools**: sandboxed shell (allow-list), sandboxed filesystem, HTTP, and a
  bridge to the Rust perf service.
- **Streaming chat** end-to-end (Go gateway → Python core → NIM).
- **Rust** module exposes parallel cosine similarity (`rayon`) and CRC32 over
  HTTP for the Python core to call.
- **20 Python tests + 5 Rust tests + Go vet/build** all green.

---

## Project Structure

```
.
├── ai_core/                 Python AI core (FastAPI + agents)
│   ├── agents/              Planner / Executor / Coder / Critic
│   ├── orchestrator/        autonomous loop
│   ├── memory/              short-term + FAISS long-term
│   ├── tools/               shell, fs, http, rust_perf
│   ├── nim_client/          NIM client + router
│   ├── config.py            pydantic settings
│   ├── logging_setup.py     loguru setup
│   ├── main.py              FastAPI app
│   └── requirements.txt
├── api-gateway/             Go gateway (Fiber)
│   ├── main.go
│   ├── routes/              proxy + streaming
│   └── middleware/          request-id + rate limit
├── performance/             Rust crate
│   ├── src/lib.rs           cosine / normalize / crc32 (rayon)
│   └── src/main.rs          axum HTTP server (perf-server)
├── config/
│   ├── models.yaml          model routing tiers
│   └── settings.yaml        agents / memory / tools
├── tests/                   pytest (offline, fully mocked)
├── scripts/run_all.sh       boots the full stack
├── .env.example
└── README.md
```

> **Why `ai_core/` (underscore)?** Python cannot import directories with
> hyphens. The package name is `ai_core`; everything else still uses the
> hyphenated diagram in docs/architecture.

---

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# edit .env and set NVIDIA_NIM_API_KEY=nvapi-...
```

Get a key at <https://build.nvidia.com/> → "Get API Key".

### 2. Boot everything

```bash
./scripts/run_all.sh
```

This launches:

- Rust **perf-server** on `:7070`
- Python **AI core** on `:8000`
- Go **API gateway** on `:9000`

Logs land in `./logs/`. Stop with `Ctrl+C` (or `STOP=1 ./scripts/run_all.sh`).

### 3. Use it

```bash
# Health
curl http://localhost:9000/health

# Direct chat (auto-routed to fast tier)
curl -X POST http://localhost:9000/api/v1/chat \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Autonomous agent loop
curl -X POST http://localhost:9000/api/v1/agent/run \
     -H 'Content-Type: application/json' \
     -d '{"goal":"Plan a 3-step launch checklist for a SaaS MVP."}'

# Streaming
curl -N -X POST http://localhost:9000/api/v1/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Tell me a joke"}],"tier":"fast"}'

# Memory
curl -X POST http://localhost:9000/api/v1/memory/store \
     -H 'Content-Type: application/json' \
     -d '{"text":"User prefers concise answers.","tags":["preference"]}'

curl -X POST http://localhost:9000/api/v1/memory/search \
     -H 'Content-Type: application/json' \
     -d '{"query":"user preferences","k":3}'

# Tools
curl http://localhost:9000/api/v1/tools
curl -X POST http://localhost:9000/api/v1/tools/run \
     -H 'Content-Type: application/json' \
     -d '{"name":"shell","args":{"command":"echo hello"}}'
```

---

## Running Components Individually

### Python AI core

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai_core/requirements.txt
python -m ai_core            # → :8000
```

### Go gateway

```bash
cd api-gateway
go mod tidy
go run .                     # → :9000
```

### Rust perf service

```bash
cd performance
cargo run --release --bin perf-server   # → :7070
```

---

## Model Routing

Edit `config/models.yaml`:

```yaml
routing:
  complex:  { model: meta/llama-3.1-70b-instruct,        temperature: 0.4 }
  balanced: { model: mistralai/mixtral-8x7b-instruct-v0.1, temperature: 0.5 }
  fast:     { model: mistralai/mistral-7b-instruct-v0.3,   temperature: 0.6 }
  code:     { model: meta/llama-3.1-70b-instruct,        temperature: 0.2 }
```

The router picks a tier from:

1. Explicit `tier` in the request (`"complex" | "balanced" | "fast" | "code"`).
2. Heuristic keyword match (`code`, `design`, `analyze`, …).
3. Prompt length (>1200 chars → complex).
4. Default → `balanced`.

---

## Agent Loop

```
goal → planner.plan → executor.execute_plan → critic.review
                                                  │
                                            score < threshold
                                                  │
                                          refine goal & retry
                                                  │
                                          score ≥ threshold
                                                  │
                                                final
```

- `MAX_ITERATIONS` (default 5) and `CRITIC_THRESHOLD` (default 0.75) are
  configurable via `.env`.
- Every Q/A pair is persisted to long-term memory so future runs benefit
  from accumulated context.

---

## Testing

```bash
# Python
source .venv/bin/activate
python -m pytest tests/ -v

# Rust
cd performance && cargo test --release

# Go
cd api-gateway && go vet ./... && go build ./...
```

All Python tests are **offline** — they stub the NIM client so CI works
without an API key.

---

## API Reference (Gateway)

| Method | Path                       | Notes                                    |
| ------ | -------------------------- | ---------------------------------------- |
| GET    | `/`                        | service metadata                         |
| GET    | `/health`                  | gateway + core health                    |
| POST   | `/api/v1/chat`             | one-shot chat completion                 |
| POST   | `/api/v1/chat/stream`      | streamed chat (text/plain)               |
| POST   | `/api/v1/agent/run`        | autonomous Plan → Execute → Critic loop  |
| POST   | `/api/v1/memory/store`     | embed + persist text                     |
| POST   | `/api/v1/memory/search`    | top-k vector search                      |
| GET    | `/api/v1/tools`            | list registered tools + JSON schemas     |
| POST   | `/api/v1/tools/run`        | invoke a tool                            |

Rate limit: 60 req/min/IP (configurable in `api-gateway/main.go`).

---

## Configuration

| Variable                 | Default                           | Purpose                          |
| ------------------------ | --------------------------------- | -------------------------------- |
| `NVIDIA_NIM_API_KEY`     | _(required)_                      | NIM bearer token                 |
| `NVIDIA_NIM_BASE_URL`    | `https://integrate.api.nvidia.com/v1` | NIM endpoint                 |
| `AI_CORE_HOST` / `_PORT` | `0.0.0.0:8000`                    | Python core bind                 |
| `GATEWAY_HOST` / `_PORT` | `0.0.0.0:9000`                    | Go gateway bind                  |
| `AI_CORE_URL`            | `http://localhost:8000`           | gateway → core upstream          |
| `RUST_PERF_URL`          | `http://localhost:7070`           | core → rust upstream             |
| `MAX_ITERATIONS`         | `5`                               | agent loop cap                   |
| `CRITIC_THRESHOLD`       | `0.75`                            | accept score                     |
| `MEMORY_DIR`             | `./data/memory`                   | FAISS persist dir                |
| `SHORT_TERM_MAX`         | `50`                              | per-session FIFO size            |
| `LONG_TERM_DIM`          | `1024`                            | embedding dimension              |

---

## Security Notes

- **Shell tool**: argv-only execution (no `shell=True`), enforced allow-list.
- **Filesystem tool**: every path resolved against a sandbox dir; traversal
  rejected.
- **Gateway**: per-IP token-bucket rate limiter, request-id middleware.
- **NIM key** never leaves the Python core; the gateway is auth-passthrough.

---

## License

MIT — do whatever you want, no warranty.
