---
title: AI Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# AI Agent — Hybrid RAG + SQL Assistant

An end-to-end AI assistant that routes natural-language questions between two retrieval pipelines: a **RAG pipeline** over PDFs (using FAISS + sentence-transformer embeddings) and a **SQL pipeline** over a SQLite database. An LLM-based intent router decides which pipeline handles each query. Deployed as a FastAPI service on HuggingFace Spaces.

**Live demo:** [amogh781-ai-agent.hf.space]([https://amogh781-ai-agent.hf.space](https://huggingface.co/spaces/amogh781/AI_agent_UI)) · **API docs:** [/docs](https://amogh781-ai-agent.hf.space/docs)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (POST /query)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   LLM-Based Router  │  ← classifies as SQL or RAG
                  │   (router.py)       │
                  └──────┬───────┬──────┘
                         │       │
              SQL ◄──────┘       └──────► RAG
                │                           │
   ┌────────────▼─────────┐    ┌────────────▼─────────────┐
   │   sql_module.py      │    │   rag_module.py          │
   │  • LLM → SQL gen     │    │  • FAISS retrieval (k=2) │
   │  • SELECT-only guard │    │  • Conversational memory │
   │  • Read-only DB conn │    │  • Groq llama-3.1-8b     │
   └────────────┬─────────┘    └────────────┬─────────────┘
                │                            │
                ▼                            ▼
         SQLite (movies.db)         FAISS index (PDFs)
```

## Features

- **Hybrid retrieval** — single API endpoint routes between structured (SQL) and unstructured (RAG) data sources based on query intent
- **Conversational memory** — session-scoped windowed memory (`ConversationBufferWindowMemory`) keyed by `session_id`, retaining the last 5 exchanges per user
- **SQL safety** — defense-in-depth against LLM-generated destructive SQL: an allowlist parser that accepts only `SELECT` statements, combined with a read-only SQLite connection
- **Batched embedding** — FAISS index built in batches of 64 chunks to stay within memory limits on free-tier hardware while indexing the full document corpus
- **Lazy model loading** — embedding model, FAISS index, and LLM clients load on first request, keeping cold-start memory low

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Uvicorn, Pydantic |
| **LLM** | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector store** | FAISS (local, CPU) |
| **Document loading** | `PyPDFLoader`, `RecursiveCharacterTextSplitter` |
| **Memory & chains** | LangChain (`ConversationalRetrievalChain`, `ConversationBufferWindowMemory`) |
| **Database** | SQLite (read-only mode for query execution) |
| **Deployment** | Docker on HuggingFace Spaces |

## Project Structure

```
ai-agent-project/
├── src/
│   ├── main.py            ← FastAPI entry point, /query and /reset endpoints
│   ├── router.py          ← LLM-based intent classifier (SQL vs RAG)
│   ├── rag_module.py      ← RAG pipeline with conversational memory
│   ├── rag_pipeline.py    ← Builds the FAISS index from PDFs (batched)
│   ├── sql_module.py      ← Text-to-SQL with safety guardrails
│   ├── database.py        ← SQLite setup utilities
│   └── data_loader.py     ← Data cleaning helpers
├── pdf_files/             ← Source PDFs for RAG indexing
├── faiss_index/           ← Pre-built FAISS index (committed for fast startup)
├── data/
│   └── movies.db          ← SQLite database
├── Dockerfile             ← Container definition for HF Spaces deployment
├── requirements.txt
└── start.sh
```

## API Reference

### `POST /query`

Submit a natural-language question. The router decides whether to use RAG or SQL.

**Request body:**
```json
{
  "question": "Compare Counter-Strike 2 and Valorant",
  "session_id": "user_alice"
}
```

`session_id` is optional (defaults to `"default"`). Pass a unique value per user/session to isolate conversational memory.

**Response:**
```json
{
  "response": "Counter-Strike 2 and Valorant are two popular...",
  "route": "RAG"
}
```

### `POST /reset`

Clear conversational memory for a session.

**Request body:**
```json
{ "session_id": "user_alice" }
```

### `GET /`

Health check endpoint.

## Running Locally

### Prerequisites
- Python 3.10+
- A Groq API key (free at [console.groq.com/keys](https://console.groq.com/keys))

### Setup

```bash
# Clone and enter the repo
git clone https://github.com/chippad781/ai-agent-project.git
cd ai-agent-project

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="your_key_here"      # macOS/Linux
set GROQ_API_KEY=your_key_here           # Windows CMD

# Build the FAISS index from PDFs in pdf_files/
python -m src.rag_pipeline

# Start the API
uvicorn src.main:app --host 0.0.0.0 --port 7860
```

The API is now live at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

### Example requests

```bash
# RAG query with session memory
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare Counter-Strike 2 and Valorant", "session_id": "alice"}'

# Follow-up using conversational memory
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which one is more popular?", "session_id": "alice"}'

# SQL query against the database
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many movies were released after 2015?"}'
```

## Design Decisions

**Why LLM-based routing instead of keyword rules?**
Keyword rules break on natural-language variation ("how many" vs "count" vs "tell me the number of"). LLM routing handles paraphrasing for free at the cost of one extra API call per query. A future iteration would cache routing decisions and fine-tune a small classifier (e.g., DistilBERT) to eliminate the routing call entirely.

**Why FAISS over a managed vector database (Pinecone, Weaviate, Qdrant)?**
FAISS runs locally with no service dependency, which is the right choice for a free-tier deployment. Tradeoff: no built-in metadata filtering at scale, no managed scaling, and re-indexing requires rebuilding. For a multi-tenant production app I'd migrate to Qdrant (self-hosted) or Pinecone (managed).

**Why MiniLM-L6-v2 over larger embedding models?**
384-dim embeddings are fast on CPU and good enough for the document scale here. For a production app with quality-sensitive retrieval, I'd switch to `bge-large-en-v1.5` (self-hosted) or OpenAI's `text-embedding-3-small`.

**Why `k=2` retrieval with no reranker?**
Keeps the context window small and reduces noise from semantically-adjacent-but-irrelevant chunks. The obvious upgrade is retrieving `k=20` candidates with FAISS and reranking with a cross-encoder (BGE-reranker or Cohere Rerank) to pick the top 3 — better quality at the cost of ~100-300ms additional latency.

**SQL safety strategy.**
LLM-generated SQL runs against an SQLite database opened in read-only mode via the `?mode=ro` URI parameter, with an allowlist parser rejecting anything that isn't a single `SELECT` statement. Defense in depth: even if the allowlist somehow misses a destructive query, the read-only connection prevents it from executing.

## Roadmap / Future Work

- [ ] **Hybrid retrieval** — combine BM25 (sparse) with FAISS (dense) using reciprocal rank fusion
- [ ] **Reranking layer** — BGE-reranker on top-k FAISS candidates for better precision
- [ ] **RAGAS eval framework** — automated faithfulness and context-relevance scoring on a fixed eval set, run on every code change
- [ ] **Migrate to LangGraph** — explicit state-machine control flow for the agent, easier debugging than chained calls
- [ ] **Multi-provider LLM failover** — secondary provider (OpenAI/Anthropic) as fallback when Groq is unavailable
- [ ] **Authentication and rate limiting** — JWT-based auth with per-user rate limits via Redis token bucket
- [ ] **Observability** — Prometheus metrics for latency (p50/p95/p99), error rate, LLM cost per query
- [ ] **Hybrid SQL/RAG routing** — fan out to both pipelines on ambiguous queries and merge results

## Deployment Notes

Deployed on HuggingFace Spaces (Docker SDK). The free tier sleeps after 48 hours of inactivity; an UptimeRobot HTTP monitor pings the endpoint every 5 minutes to keep it warm. For real production traffic, the Space would move to HF persistent hardware or a paid container service (Render, Railway, AWS Fargate) with `min_instances=1`.

## License

MIT.
