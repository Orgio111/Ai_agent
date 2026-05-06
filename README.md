# JARVIS — Autonomous AI Operating System

> **Production-grade, event-driven AI OS**: 3-tier multi-model swarm, persistent memory, GPU+CPU dual scheduling, voice interface, autonomous goal DAG, and self-improving loop — powered by NVIDIA NIM, OpenRouter free swarm, and ranked fallbacks.

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
 │  Core    │ │  Service   │ │  GPU+CPU Scheduler    │
 │  (8000)  │ │  (8003)    │ │  Circuit Breaker      │
 │          │ │            │ │  Model Router         │
 │ Planner  │ │ Working    │ │  Streaming            │
 │ Executor │ │ Episodic   │ └──────────────────────┘
 │ Critic   │ │ Semantic   │
 │ Research │ │ Procedural │ ┌──────────────────────┐
 │ Optimize │ │ Decay      │ │  Tool System (8004)  │
 │          │ │ FAISS/GPU  │ │  Shell · FS · HTTP   │
 │ Lifecycle│ │ Redis      │ │  Code · Search       │
 │ Goal DAG │ │ PostgreSQL │ │  Sandbox · Perms     │
 └──────────┘ └────────────┘ └──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                  3-Tier Model Priority System                │
│                                                              │
│  Tier 1: NVIDIA NIM (primary, 5 specialized models)         │
│     ↓ (unavailable / overloaded)                            │
│  Tier 2: OpenRouter Free Swarm (14 models, parallel fan-out)│
│     ↓ (all free models exhausted)                           │
│  Tier 3: Ranked Paid Fallbacks (7 models, best-fit)         │
└──────────────────────────────────────────────────────────────┘
```

### Request Flow
```
User → Task Classifier (<1ms regex) → TaskType
  → ModelSelector (tier 1→2→3)   → best model(s)
  → Specialized Agent             → NIM / OpenRouter / Ranked
  → MultiModelExecutor (optional) → N parallel models → aggregate
  → Auditor                       → quality score
  → SelfImprovingLoop.record()    → EMA tracking
  → Memory store (async)          → Response → UI
```

---

## 3-Tier Model Priority System

### Tier 1 — NVIDIA NIM (Primary)

Five role-specialized models, overridable via environment variables:

| Role | Default Model | Env Override |
|---|---|---|
| Planner / General | `nvidia/llama-3.1-nemotron-ultra-253b-v1` | `NIM_MODEL_PLANNER` |
| Perception / Vision | `nvidia/nemotron-mini-4b-instruct` | `NIM_MODEL_PERCEPTION` |
| Research / Reasoning | `deepseek-ai/deepseek-r1` | `NIM_MODEL_RESEARCHER` |
| Audit / Validation | `meta/llama-3.1-405b-instruct` | `NIM_MODEL_AUDITOR` |
| Engineering / Code | `qwen/qwen2.5-72b-instruct` | `NIM_MODEL_ENGINEER` |

### Tier 2 — OpenRouter Free Swarm

14 free models grouped by capability. Used when NIM is unavailable or when `select_swarm()` fans out for parallel consensus:

| Capability Group | Models |
|---|---|
| Reasoning | `hunyuan-a13b`, `qwen3-235b`, `deepseek-r1-zero` |
| Coding | `qwen2.5-coder-32b`, `devstral-small`, `qwen3-30b-a3b` |
| Multimodal | `gemma-3-27b`, `llava-next-llama3`, `qwen2.5-vl-72b` |
| Reporting | `minimax-m1`, `mistral-small-3.1-24b` |
| Lightweight | `llama-3.3-70b`, `llama-3.2-3b`, `gemma-3-12b` |

### Tier 3 — Ranked Paid Fallbacks

7 models ranked by task fitness, used as last resort:
`claude-sonnet-4-5`, `mistral-large`, `qwen2.5-plus`, `minimax-m1`, `deepseek-chat`, `hunyuan`, `kimi-k2`

### Model Availability Blacklist

Failed models are automatically blacklisted for **60 seconds** and retried after cooldown. The `SelfImprovingLoop` further deprioritizes models whose rolling average quality score drops below **0.5**.

---

## Specialized Agents

### PerceptionAgent
- **Primary model**: Nemotron Nano Omni (multimodal)
- Handles: image/document parsing, chart analysis, audio transcription
- Routes to multimodal-capable models when image content is detected

### ResearchAgent
- **Primary model**: DeepSeek R1 (chain-of-thought reasoning)
- Decomposes queries into 3–5 sub-questions, executes in parallel, synthesizes
- Output: structured JSON with `summary`, `hypotheses`, `confidence`, `gaps`, `strategy_recommendation`

### EngineerAgent
- **Primary model**: GLM/Qwen Engineering
- ReAct loop: Reason → Generate → Execute → Observe → Self-correct (up to 3 retries)
- Supports: code generation, sandboxed execution, financial backtesting

### AuditAgent
- **Primary model**: GPT-OSS-120B (high-accuracy validation)
- Returns structured JSON: `pass`, `score`, `issues`, `hallucinations`, `risks`, `recommendations`, `revised_output`
- Methods: `audit()`, `validate_code()`, `detect_hallucinations()`, `risk_assess()`

### ReporterAgent
- **Primary model**: MiniMax (communication specialist)
- Formats: executive summary, CFO financial report, CTO technical report
- Audiences: board, CFO, CTO, general
- Output structure: Executive Summary → Key Findings → Analysis → Risks → Recommendations

---

## Multi-Model Parallel Executor

`MultiModelExecutor.run(messages, task, n_models=3)` fans out a single prompt to N models concurrently via `asyncio.gather`, then aggregates:

- **Consensus**: if all models return identical content, return it once
- **Divergent**: return the highest word-count response (most complete)
- Failed models are auto-blacklisted; the executor selects diverse models across tiers

```python
from ai_core.multi_model import get_executor
from ai_core.model_selector import TaskType

executor = get_executor()
result = await executor.run(messages, task=TaskType.RESEARCH, n_models=3)
print(result.consensus)      # aggregated answer
print(result.best.model_id)  # best single response
print(result.ok_count)       # how many succeeded
```

---

## Task Classifier

Sub-millisecond keyword routing — no LLM call needed:

```python
from ai_core.orchestrator.task_classifier import get_classifier

classifier = get_classifier()
task_type = classifier.classify("analyze this chart and summarize findings")
# → TaskType.PERCEPTION or TaskType.REPORTING depending on keyword weight
```

8 task types: `PERCEPTION`, `RESEARCH`, `ENGINEERING`, `AUDIT`, `REPORTING`, `PLANNING`, `REASONING`, `GENERAL`

---

## Deep Search Engine

```python
from ai_core.search import get_search_engine

engine = get_search_engine()
result = await engine.search(query, session_context=ctx, require_fresh=True)
print(result.synthesis)        # synthesized answer
print(result.confidence)       # 0.0–1.0
print(result.contradictions)   # cross-source conflicts detected
```

Decomposes queries into sub-queries, executes in parallel, detects contradictions, returns confidence-scored synthesis.

---

## Autonomous Goal System

Persistent DAG-backed goal queue. Goals survive restarts and run unattended for 8+ hours.

```bash
# Create a long-running goal
curl -X POST http://localhost:8000/goals \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Research quantum computing advances and write a CFO-ready summary",
    "priority": "HIGH",
    "deadline": "2025-12-31T23:59:59"
  }'

# Poll status
curl http://localhost:8000/goals/<goal_id>
```

Goal lifecycle: `PENDING → ACTIVE → COMPLETED` (auto-retry up to 3× on failure, `BLOCKED` if dependencies unmet).

```python
from ai_core.goals import AutonomousGoalSystem, GoalPriority

goal_system = AutonomousGoalSystem(persist_path=Path("goals.json"))
await goal_system.start()

goal_id = await goal_system.add_goal(
    description="Analyze market trends for Q4",
    priority=GoalPriority.HIGH,
    depends_on=[],
)
sub_ids = await goal_system.decompose(goal_id, sub_descriptions=[...])
```

---

## Self-Improving Loop

Continuously monitors model performance and adapts routing:

```python
from ai_core.self_improve.loop import SelfImprovingLoop

loop = SelfImprovingLoop(persist_path=Path("perf.json"), improvement_interval=300.0)
await loop.start()  # background analysis every 5 minutes

# After each inference:
loop.record(model_id="deepseek-r1", task_type=TaskType.RESEARCH,
            quality_score=0.91, latency_ms=1240.0)

# Get ranked models for a task:
ranking = loop.get_model_ranking(TaskType.RESEARCH)
# → [{"model_id": ..., "avg_quality": 0.91, "composite_score": 0.87, ...}]

# System stats:
print(loop.stats())
```

**Improvement axes:**
1. Routing — deprioritize models with avg quality < 0.5
2. Patterns — prefer models with avg quality > 0.85 for specific task types
3. Persistence — last 1,000 records + 200 insights saved to JSON

---

## GPU + CPU Dual Scheduling

The LLM Engine schedules inference across **both GPU and CPU simultaneously**:

- **GPU** handles LLM inference (large models requiring VRAM)
- **CPU worker pool** handles lightweight workloads (embeddings, fast models)
- `prefer_cpu=True` routes embeddings to CPU, keeping GPU free for concurrent LLM calls
- FAISS index is accelerated on GPU when available; CPU copy kept in sync for persistence

```
GPUScheduler
├── GPU devices  [device_0: A100 80GB, device_1: A100 80GB, ...]
│     allocate(model_id, required_mem_gb=40.0)  → AllocationHandle(device_type="gpu")
└── CPUWorkerPool [total_workers=os.cpu_count()]
      allocate(model_id, prefer_cpu=True)        → AllocationHandle(device_type="cpu")
```

Status endpoint includes both resource pools:
```json
{
  "gpu_count": 2,
  "devices": [{"id": 0, "free_gb": 38.2, "utilization": "52%"}, ...],
  "cpu_pool": {"total_workers": 16, "active_workers": 3, "utilization_pct": 18.75}
}
```

---

## Memory Layers

| Layer | Backend | Retention | Purpose |
|---|---|---|---|
| **Working** | Redis | Session TTL | Active conversation context |
| **Episodic** | PostgreSQL + FAISS | 72h + Ebbinghaus decay | Past events and interactions |
| **Semantic** | PostgreSQL + FAISS | Permanent | Deduplicated concept knowledge |
| **Procedural** | PostgreSQL | Permanent | Learned action patterns + success rates |

**Pipeline**: `query → embed (NIM/CPU) → FAISS GPU search → metadata filter → rerank → [summarize] → response`

Memory stores are non-blocking: `store()` uses `asyncio.to_thread()` for FAISS I/O and `asyncio.create_task()` for fire-and-forget background writes.

---

## Services

| Service | Port | Stack | Purpose |
|---|---|---|---|
| **Agent Core** | 8000 | Python/FastAPI | Multi-agent swarm + Autonomous Goal Engine |
| **Broker** | 8001 | Rust/Tokio | Priority event bus, Redis DLQ, WebSocket |
| **LLM Engine** | 8002 | Python/FastAPI | NIM + GPU/CPU scheduling + circuit breaker |
| **Memory** | 8003 | Python/FastAPI | Redis + PostgreSQL + FAISS memory pipeline |
| **Tool System** | 8004 | Python/FastAPI | Sandboxed tool execution with RBAC |
| **Voice** | 8005 | Python/FastAPI | Whisper STT + edge-tts TTS + WebRTC VAD |
| **UI** | 3000 | React/TypeScript | Chat, agent chain viz, goal manager |
| **Prometheus** | 9090 | Prometheus | Metrics scraping |
| **Grafana** | 3001 | Grafana | Dashboards |
| **OTel Collector** | 4317 | OpenTelemetry | Distributed tracing |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker + Docker Compose | ≥ 27 |
| **NVIDIA NIM API Key** | [build.nvidia.com](https://build.nvidia.com) |
| **OpenRouter API Key** | [openrouter.ai](https://openrouter.ai) |
| Python | 3.12+ (local dev) |
| Rust | 1.82+ (broker) |
| Node.js | 20+ (UI) |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Orgio111/ai_agent.git && cd ai_agent
cp .env.example .env
# Edit .env — required keys:
#   NVIDIA_NIM_API_KEY=nvapi-...
#   OPENROUTER_API_KEY=sk-or-...

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

---

## Environment Variables

### Required
| Variable | Description |
|---|---|
| `NVIDIA_NIM_API_KEY` | NVIDIA NIM API key (primary LLM tier) |
| `OPENROUTER_API_KEY` | OpenRouter API key (free swarm + ranked fallbacks) |

### Optional — NIM Model Overrides
| Variable | Default | Description |
|---|---|---|
| `NIM_MODEL_PLANNER` | `nvidia/llama-3.1-nemotron-ultra-253b-v1` | Planning and general reasoning |
| `NIM_MODEL_PERCEPTION` | `nvidia/nemotron-mini-4b-instruct` | Vision and document parsing |
| `NIM_MODEL_RESEARCHER` | `deepseek-ai/deepseek-r1` | Deep research and hypothesis generation |
| `NIM_MODEL_AUDITOR` | `meta/llama-3.1-405b-instruct` | Quality audit and hallucination detection |
| `NIM_MODEL_ENGINEER` | `qwen/qwen2.5-72b-instruct` | Code generation and backtesting |

### Optional — Infrastructure
| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM API base URL |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Voice Interface

WebSocket at `ws://localhost:8005/ws/voice`:

```json
{"action": "audio",    "data": "<base64 PCM>"}
{"action": "tts_only", "text": "Hello JARVIS"}
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
{"name": "my_tool", "endpoint": "http://my-svc/execute", "required_role": "agent"}
```

---

## Security

- **SSRF protection**: `http_get` blocks all RFC-1918, loopback, and link-local addresses
- **Allowlist enforcement**: shell tool rejects any unlisted command
- **Path traversal protection**: filesystem tool enforces sandbox root via `Path.resolve()`
- **Static code analysis**: code execution tool blocks `os.system`, `subprocess`, `socket`, etc.
- **RBAC**: 4-tier hierarchy (admin → operator → agent → readonly) with HMAC-signed tokens
- **Circuit breaker**: opens after 5 NIM failures, auto-recovers after 30s (lock released before coroutine to prevent cascading latency)
- **Connection pooling**: httpx client with `max_connections=100`, `max_keepalive_connections=50`

---

## Observability

| Metric | Description |
|---|---|
| `jarvis_broker_events_published_total` | Events by topic |
| `jarvis_broker_queue_depth` | Live queue depth |
| `jarvis_broker_dlq_depth` | Dead-letter queue size |
| `jarvis_llm_latency_seconds` | Histogram by model/tier |
| `jarvis_llm_tokens_total` | Token usage by model |
| `jarvis_model_quality_ema` | Per-model rolling quality score |
| `jarvis_gpu_utilization` | GPU memory allocation per device |
| `jarvis_cpu_pool_utilization` | CPU worker pool saturation |

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
kubectl -n jarvis edit secret jarvis-secrets
# set NVIDIA_NIM_API_KEY and OPENROUTER_API_KEY

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
├── .github/workflows/        ci.yml · deploy.yml
├── services/
│   ├── broker/               Rust Tokio event broker (priority queues, DLQ, WS)
│   ├── llm-engine/           FastAPI LLM wrapper (GPU+CPU scheduler, circuit breaker)
│   │   └── src/
│   │       ├── gpu_scheduler.py   GPU + CPUWorkerPool dual scheduling
│   │       ├── circuit_breaker.py Lock-free circuit breaker (lock released before await)
│   │       ├── engine.py          Inference engine (embed→CPU, LLM→GPU)
│   │       └── model_router.py    Tier-based model routing
│   ├── memory/               Multi-layer memory (Redis + PG + FAISS GPU + decay)
│   ├── agent-core/           Agent swarm + DAG executor + parallel task levels
│   │   └── src/
│   │       └── swarm.py           Parallel DAG execution via graphlib + asyncio.gather
│   ├── tool-system/          Sandboxed tools with RBAC and failure-aware ranking
│   ├── voice/                Whisper STT + edge-tts + WebRTC VAD
│   ├── ui/                   React UI + Tauri desktop shell
│   └── observability/        Prometheus + OTel Collector configs
├── ai_core/                  Autonomous AI core (Python)
│   ├── agents/
│   │   ├── perception.py     PerceptionAgent — multimodal, document/chart parsing
│   │   ├── researcher.py     ResearchAgent — DeepSeek R1, parallel sub-query research
│   │   ├── engineer.py       EngineerAgent — ReAct loop, code gen, backtesting
│   │   ├── auditor.py        AuditAgent — hallucination detection, risk assessment
│   │   ├── reporter.py       ReporterAgent — CFO/CTO/board reports in markdown
│   │   ├── planner.py        Planner — goal → JSON task DAG
│   │   ├── executor.py       SmartExecutor — parallel tool dispatch
│   │   └── critic.py         Critic — quality scoring (threshold 0.7)
│   ├── model_selector/
│   │   ├── capability.py     TaskType (8) + Capability (12) enums + UnifiedModelSpec
│   │   ├── nim_models.py     5 NIM role-specialized models with env-var overrides
│   │   ├── ranking.py        7 ranked paid fallback models
│   │   └── selector.py       3-tier selector + select_swarm() + 60s blacklist
│   ├── openrouter/
│   │   ├── client.py         OpenRouterClient — httpx pool, retry, streaming
│   │   └── models.py         14 free models grouped by capability
│   ├── multi_model/
│   │   └── executor.py       Fan-out to N models in parallel, aggregate consensus
│   ├── search/
│   │   └── engine.py         DeepSearchEngine — sub-query decomposition, contradiction detection
│   ├── goals/
│   │   └── system.py         AutonomousGoalSystem — persistent JSON DAG, 8h+ tasks
│   ├── self_improve/
│   │   └── loop.py           SelfImprovingLoop — EMA quality tracking, auto-deprioritization
│   ├── orchestrator/
│   │   ├── orchestrator.py   Main orchestrator — classifier → agent → audit → record
│   │   └── task_classifier.py <1ms regex task routing (no LLM call)
│   ├── memory/
│   │   ├── manager.py        Async memory manager (asyncio.to_thread for FAISS I/O)
│   │   ├── long_term.py      FAISS index with GPU acceleration + CPU sync copy
│   │   └── short_term.py     Redis working memory
│   ├── nim_client/           NIM API client with connection pooling
│   └── config.py             Pydantic settings (NIM key, OpenRouter key, overrides)
├── deploy/
│   ├── docker-compose.yml
│   └── kubernetes/           Deployments + Services + HPAs for all services
├── tests/                    Unit · integration · stress · e2e tests
└── scripts/                  run_all.sh · test_all.sh
```

---

## License

MIT
