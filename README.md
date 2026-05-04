# ⚡ JARVIS — Autonomous AI Operating System

> **JARVIS-class event-driven AI OS**: multi-agent swarm, persistent memory, GPU scheduling, voice interface, self-improving loop, and full production deployment — powered exclusively by NVIDIA NIM APIs.

---

## Architecture Overview

```
User / External Event
        │
        ▼
┌───────────────────┐
│  React UI / Voice │  ← Tauri desktop + WebSocket + STT/TTS
└────────┬──────────┘
         │ WebSocket / HTTP
         ▼
┌────────────────────────────────────────────────────────────┐
│              Rust Tokio Message Broker (port 8001)         │
│  Priority queues · Event sourcing · Redis-backed DLQ       │
│  WebSocket multiplexing · Backpressure · Cancellation      │
└──────┬──────────┬──────────────┬─────────────────────────┘
       │          │              │
       ▼          ▼              ▼
 ┌──────────┐ ┌────────────┐ ┌──────────────────────┐
 │  Agent   │ │  Memory    │ │  LLM Engine (8002)   │
 │  Core    │ │  Service   │ │  GPU Scheduler        │
 │  (8000)  │ │  (8003)    │ │  Circuit Breaker      │
 │          │ │            │ │  Model Router         │
 │ Planner  │ │ Working    │ │  Streaming            │
 │ Executor │ │ Episodic   │ └──────────────────────┘
 │ Critic   │ │ Semantic   │
 │ Research │ │ Procedural │ ┌──────────────────────┐
 │ Optimize │ │ Decay      │ │  Tool System (8004)  │
 │          │ │ FAISS      │ │  Shell · FS · HTTP   │
 │ Lifecycle│ │ Redis      │ │  Code · Search       │
 │ Goal DAG │ │ PostgreSQL │ │  Sandbox · Perms     │
 └──────────┘ └────────────┘ └──────────────────────┘
```

### Event Flow
```
User → Broker (priority queue) → Agent Core
  → Planner (complex model)   → Task DAG
  → Executor × N (parallel)   → Tool dispatch
  → Researcher                → Semantic memory
  → Critic (highest priority) → score ≥ 0.7?
  → Optimizer (if failed)     → learned rules
  → Memory store              → Response → UI
```

---

## Services

| Service | Port | Stack | Purpose |
|---|---|---|---|
| **Agent Core** | 8000 | Python/FastAPI | Multi-agent swarm + Goal Engine |
| **Broker** | 8001 | Rust/Tokio | Priority event bus, Redis DLQ, WebSocket |
| **LLM Engine** | 8002 | Python/FastAPI | NIM API + GPU scheduling + circuit breaker |
| **Memory** | 8003 | Python/FastAPI | Redis + PostgreSQL + FAISS memory pipeline |
| **Tool System** | 8004 | Python/FastAPI | Sandboxed tool execution with RBAC |
| **Voice** | 8005 | Python/FastAPI | Whisper STT + edge-tts TTS + WebRTC VAD |
| **UI** | 3000 | React/TypeScript | Chat, agent chain viz, goal manager |
| **Prometheus** | 9090 | Prometheus | Metrics scraping |
| **Grafana** | 3001 | Grafana | Dashboards (admin / jarvis) |
| **OTel Collector** | 4317 | OpenTelemetry | Distributed tracing |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker + Docker Compose | ≥ 27 |
| **NVIDIA NIM API Key** | [build.nvidia.com](https://build.nvidia.com) |
| Python | 3.12+ (local dev) |
| Rust | 1.82+ (broker) |
| Node.js | 20+ (UI) |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Orgio111/ai_agent.git && cd ai_agent
cp .env.example .env
#  → Edit .env: set NVIDIA_NIM_API_KEY=nvapi-your-real-key

# 2. Launch full stack
./scripts/run_all.sh

# 3. Start the UI
cd services/ui && npm install && npm run dev
# → Open http://localhost:3000

# 4. Verify
curl http://localhost:8000/health   # Agent Core
curl http://localhost:8001/health   # Broker
curl http://localhost:8002/health   # LLM Engine
```

> **The system will refuse to start** if `NVIDIA_NIM_API_KEY` is missing or is the placeholder. No mock/fallback LLM mode exists — this is intentional.

---

## Agents

### Planner
Decomposes goals into structured JSON task DAGs. Routes to `complex` tier (llama-3.1-70b). Outputs tasks with dependencies, agent assignments, and priority scores.

### SmartExecutor
Executes tasks with dynamic tool selection and parallel dispatch via `asyncio.gather`. On failure, uses an **alternative approach** rather than blind retry (max 3 attempts). Records per-tool success rates for future ranking.

### Critic *(highest broker priority)*
Scores outputs 0.0–1.0 across accuracy, completeness, safety, and clarity. Pass threshold: **0.7**. Provides revised output or triggers Optimizer. Up to 2 critique cycles per request.

### Researcher
Synthesises information via `http_get` tool. Assigns confidence levels (HIGH/MEDIUM/LOW). Stores findings as semantic memory for cross-session reuse.

### Optimizer + Self-Improving Loop
Analyses failure patterns, generates learned rules, and stores them as procedural memory. `SelfImprovingLoop` auto-triggers after 3 repeated failures on the same pattern.

---

## Memory Layers

| Layer | Backend | Retention | Purpose |
|---|---|---|---|
| **Working** | Redis | Session TTL | Active conversation context |
| **Episodic** | PostgreSQL + FAISS | 72h + Ebbinghaus decay | Past events and interactions |
| **Semantic** | PostgreSQL + FAISS | Permanent | Deduplicated concept knowledge |
| **Procedural** | PostgreSQL | Permanent | Learned action patterns + success rates |

**Pipeline**: `query → embed (NIM) → FAISS search → metadata filter → rerank → [summarize] → response`

---

## GPU Scheduling

- Discovers GPUs via `nvidia-smi` at startup; falls back to CPU transparently
- Allocates GPU memory before each inference, releases immediately after
- Refreshes memory stats every 10 seconds in background thread
- Supports MIG partitioning and multi-node cluster awareness

| Tier | Model | GPU Mem |
|---|---|---|
| `fast` | mistral-7b | 14 GB |
| `balanced` | mixtral-8x7b | 24 GB |
| `complex` | llama-3.1-70b | 40 GB |
| `code` | codellama-70b | 40 GB |
| `embed` | nv-embedqa-e5-v5 | 4 GB |

---

## Autonomous Goals

```bash
# Create a long-running autonomous goal
curl -X POST http://localhost:8000/goals \
  -H 'Content-Type: application/json' \
  -d '{"description": "Research advances in quantum computing and write a summary", "priority": 8, "auto_resume": true}'

# Poll status
curl http://localhost:8000/goals/<goal_id>
```

The Goal Engine plans → executes (DAG) → auto-resumes on failure → stores results in memory.

---

## Voice Interface

WebSocket at `ws://localhost:8005/ws/voice`:

```json
{"action": "audio",    "data": "<base64 PCM>"}   // stream mic audio
{"action": "tts_only", "text": "Hello JARVIS"}    // text → speech
{"action": "config",   "voice": "en-US-AriaNeural", "rate": "+10%"}
```

Server streams back:
```json
{"type": "stt", "text": "...", "confidence": 0.97}
{"type": "tts", "audio": "<base64 MP3>", "done": false}
```

**Latency target: ≤300ms** speech-end → first TTS byte.

---

## Tool System

| Tool | Min Role | Sandboxed |
|---|---|---|
| `shell` | operator | Yes (allowlist) |
| `filesystem` | operator | Yes (chroot) |
| `filesystem_read` | agent | Yes |
| `http_get` | agent | No (SSRF guard) |
| `code_execution` | agent | Yes (subprocess) |
| `search` | agent | No |

Register custom tools dynamically:
```bash
POST http://localhost:8004/register
{"name": "my_tool", "endpoint": "http://my-svc/execute", "required_role": "agent", ...}
```

---

## Security

- **SSRF protection**: `http_get` blocks all RFC-1918, loopback, and link-local addresses
- **Allowlist enforcement**: shell tool rejects any unlisted command
- **Path traversal protection**: filesystem tool enforces sandbox root via `Path.resolve()`
- **Static code analysis**: code execution tool blocks `os.system`, `subprocess`, `socket`, etc.
- **RBAC**: 4-tier hierarchy (admin → operator → agent → readonly) with HMAC-signed tokens
- **Circuit breaker**: opens after 5 NIM failures, auto-recovers after 30s

---

## Observability

| Metric | Description |
|---|---|
| `jarvis_broker_events_published_total` | Events by topic |
| `jarvis_broker_queue_depth` | Live queue depth |
| `jarvis_broker_dlq_depth` | Dead-letter queue size |
| `jarvis_llm_latency_seconds` | Histogram by model/tier |
| `jarvis_llm_tokens_total` | Token usage by model |

Grafana: http://localhost:3001 · Prometheus: http://localhost:9090

---

## Running Tests

```bash
pip install -r ai_core/requirements.txt pytest pytest-asyncio pytest-cov

# Unit tests (no API key needed)
pytest tests/ --ignore=tests/test_e2e_smoke.py -v

# Stress tests
pytest tests/test_stress.py -v

# Integration tests
pytest tests/test_integration.py -v

# With coverage report
pytest tests/ --ignore=tests/test_e2e_smoke.py --cov=ai_core --cov-report=html
```

```bash
# Rust broker
cd services/broker && cargo test && cargo clippy -- -D warnings
```

---

## Kubernetes Deployment

```bash
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl -n jarvis edit secret jarvis-secrets   # set real NVIDIA_NIM_API_KEY

for svc in memory broker llm-engine agent-core voice ui observability; do
  kubectl apply -f deploy/kubernetes/$svc/deployment.yaml
done

kubectl rollout status deployment/jarvis-agent-core -n jarvis
```

Label GPU nodes:
```bash
kubectl label node <node-name> accelerator=nvidia-gpu
```

All services have HPAs configured (minReplicas: 2, scales on CPU/memory).

---

## Project Structure

```
ai_agent/
├── .github/workflows/   ci.yml · deploy.yml
├── services/
│   ├── broker/          Rust Tokio event broker (priority queues, DLQ, WS)
│   ├── llm-engine/      FastAPI LLM wrapper (GPU scheduler, circuit breaker)
│   ├── memory/          Multi-layer memory (Redis + PG + FAISS + decay)
│   ├── agent-core/      Agent swarm + lifecycle manager + goal DAG engine
│   ├── tool-system/     Sandboxed tools with RBAC and failure-aware ranking
│   ├── voice/           Whisper STT + edge-tts + WebRTC VAD
│   ├── ui/              React UI + Tauri desktop shell
│   └── observability/   Prometheus + OTel Collector configs
├── deploy/
│   ├── docker-compose.yml
│   └── kubernetes/      Deployments + Services + HPAs for all services
├── ai_core/             Original agent core (Python, preserved)
├── api-gateway/         Go API gateway
├── performance/         Rust performance primitives
├── tests/               Unit · integration · stress · e2e tests
└── scripts/             run_all.sh · test_all.sh
```

---

## License

MIT
