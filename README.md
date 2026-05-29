# 🧠 ML Paper Analysis Assistant

Multi-agent system for question-answering over 5 landmark ML papers,
powered by Gemini, ChromaDB, and FastAPI.

---

## Architecture

```
User (Browser or HTTP)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI  ·  UI at /  ·  API at /api/docs   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  OrchestratorAgent                      │
│   receives question → calls tools → consolidates answer │
└──────────────┬──────────────────────────┬───────────────┘
               │ function calling         │ function calling
               ▼                          ▼
┌──────────────────────┐      ┌───────────────────────────┐
│      RAGAgent        │      │       AnalystAgent        │
│  • search_documents  │      │  • compare_papers         │
│  • extract_section   │      │  • summarize              │
│                      │      │  • rank_papers            │
└──────────┬───────────┘      └──────────────┬────────────┘
           │                                 │
           ▼                                 ▼
┌─────────────────────────────────────────────────────────┐
│    ChromaDB (Vector Store) — httpx client, no SDK       │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│         SQLite — per-thread conversation history        │
└─────────────────────────────────────────────────────────┘
```

---

## Papers Included

| # | Title | arXiv ID |
|---|---|---|
| 1 | Attention Is All You Need | 1706.03762 |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers | 1810.04805 |
| 3 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2005.11401 |
| 4 | ReAct: Synergizing Reasoning and Acting in Language Models | 2210.03629 |
| 5 | Toolformer: Language Models Can Teach Themselves to Use Tools | 2302.04761 |

Additional papers can be uploaded via the UI.

---

## Prerequisites

- **Docker Desktop** — [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)
- **Git** — [git-scm.com](https://git-scm.com/)
- **Python 3.11+** — needed only for the one-time package download

---

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/ml-paper-assistant.git
cd ml-paper-assistant
cp .env.example .env
```

Open `.env` and fill in your key:

```
GOOGLE_API_KEY=AIzaSyYOUR_REAL_KEY_HERE
```

> Get a free key at **aistudio.google.com**.
> Never commit `.env` — it is already in `.gitignore`.

### 2. Run

```bash
make setup
```

That's it. This single command:
1. Downloads Python packages to a local cache if not already present
2. Builds the Docker image from that cache — no internet needed inside Docker
3. Starts the app and ChromaDB containers
4. Ingests the 5 papers into ChromaDB

When complete:

```
✅ Setup complete!
   UI:      http://localhost:8000
   API:     http://localhost:8000/api/docs
```

---

## Commands

```bash
make setup   # first-time setup — build, start, ingest
make run     # run the 5 evaluation questions via API
make test    # run all automated tests
make down    # stop and remove containers
```

Other helpers:

```bash
make logs            # follow live app logs
make shell           # open a shell inside the app container
make ingest          # re-ingest papers after a config change
docker compose up -d # start again after make down (no rebuild needed)
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key | required |
| `GEMINI_MODEL` | LLM model for generation | `gemini-3.1-flash-lite` |
| `EMBEDDING_MODEL` | Embedding model | `gemini-embedding-001` |
| `CHROMA_HOST` | ChromaDB host (do not change) | `chromadb` |
| `CHROMA_PORT` | ChromaDB port (do not change) | `8000` |

> **Note on free tier limits:** `gemini-3.1-flash-lite` is recommended for `make run`
> as other models have lower daily limits. The evaluation run makes ~15 API calls
> total (5 questions × up to 3 tool-calling rounds each).

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/threads` | Create a new conversation thread |
| `POST` | `/threads/{id}/messages` | Send a question, get an answer |
| `GET` | `/threads/{id}/messages` | Get message history |
| `GET` | `/threads` | List all threads |
| `POST` | `/papers/upload` | Upload a new PDF paper |
| `GET` | `/health` | Health check |

---

## Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **LLM** | `gemini-3.1-flash-lite` | Best availability on free tier |
| **Embeddings** | `gemini-embedding-001` | Works with Google AI Studio free tier |
| **API versioning** | `v1beta` for LLM, `v1` for embeddings | SDK 2.7+ requires explicit versioning — tools and system instructions only work in `v1beta` |
| **Vector Store** | ChromaDB via pure httpx | Avoids `onnxruntime` crash on Docker Desktop (kernel restriction) |
| **Offline packages** | pip_cache pre-downloaded on host | Docker containers have no internet access on many machines |
| **Agent framework** | Direct SDK — no LangGraph/CrewAI | Simple 2-agent architecture, full control |
| **Persistence** | SQLite + SQLAlchemy async | Zero-dependency database for thread history |

---

## Known Limitations

1. **Rate limits** — free tier has per-minute and daily limits; `make run` adds delays between questions to stay within them
2. **No streaming** — full answer is returned after all tool calls complete
3. **PDF parsing** — `pypdf` may struggle with complex mathematical notation
4. **No authentication** — API is open, not production-ready without an auth layer
