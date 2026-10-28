# Doc-Graph-RAG: Multi-Database Document Intelligence Agent

A production-ready document ingestion and retrieval system that parses **PDF, Word, and Excel**
documents with [Docling](https://github.com/docling-project/docling), stores them in **three
complementary databases**, and answers questions through a **LangGraph agent** powered by a
configurable **OpenAI-compatible LLM** (any provider/endpoint — see [LLM backbone](#llm-backbone-any-openai-compatible-endpoint)).

## Architecture

```
                 ┌────────────────────────── Ingestion Pipeline ──────────────────────────┐
                 │                                                                          │
 PDF/Word/Excel ─┼─▶ Docling parser ─▶ Chunker ─▶ OpenAI embeddings ─▶ PostgreSQL/pgvector │
                 │        │                      │                                       │
                 │        │                      └─▶ LLM entity/relation extraction ─▶ Neo4j │
                 │        └─▶ Document & chunk metadata ─────────────────▶ PostgreSQL        │
                 └──────────────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────── LangGraph Agent ────────────────────────────┐
                 │                                                                          │
  Question ──────┼─▶ plan ─▶ retrieve (vector ∥ SQL ∥ graph, in parallel) ─▶ grade ─┐      │
                 │                                                          │ insufficient  │      │
                 │              ◀───────────────────────────────────────────┘────────┘      │
                 │                                        │ sufficient                        │
                 │                                        ▼                                  │
                 │                                   synthesize ──▶ answer + citations       │
                 └──────────────────────────────────────────────────────────────────────────┘
```

**Why three databases?**

| Database   | Role | Query type |
|------------|------|-----------|
| **PostgreSQL (relational)** | Document & chunk metadata, provenance, counts, filtering | Structured SQL |
| **PostgreSQL + pgvector** | Semantic embeddings of chunks | Similarity search |
| **Neo4j (graph)** | Entities, relations, and **cross-document links** | Multi-hop graph traversal |

## Agent strategy

The agent implements a **corrective-RAG / plan-and-execute** pattern on LangGraph:

1. **Plan** — the LLM analyzes the question and decides which retrievers to use.
2. **Retrieve** — selected retrievers run **in parallel** (semantic search, read-only SQL,
   read-only Cypher).
3. **Grade (reflect)** — the LLM assesses whether the evidence is sufficient; if not, it
   re-plans with a refined strategy (bounded retries).
4. **Synthesize** — the final answer is produced with citations back to source documents.

## Quick start

### 1. Configure

```bash
cp .env.example .env
# edit .env — set OPENAI_API_KEY at minimum
```

### 2. Start the databases

```bash
docker compose up -d
```

This starts:
- PostgreSQL 16 with pgvector (`localhost:5433`)
- Neo4j 5 community (`bolt://localhost:7687`, browser at `http://localhost:7474`)

### 3. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Initialize schemas

```bash
make init-db
```

### 5. Ingest documents

Via API:

```bash
curl -F "file=@report.pdf" http://localhost:8000/ingest/file
```

Or a whole folder (recursive):

```bash
make ingest FOLDER=./my_docs
```

### 6. Ask questions

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do document A and document B relate on the topic of X?"}'
```

Or run the agent CLI:

```bash
make ask Q="What entities link the compliance report to the budget spreadsheet?"
```

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL of **any OpenAI-compatible endpoint** (see below) |
| `OPENAI_API_KEY` | — | **Required** for embeddings + LLM (any dummy value for local servers) |
| `OPENAI_MODEL` | `gpt-4o` | Backbone LLM model name — **any model your endpoint serves** |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model served by the same endpoint |
| `EMBEDDING_DIM` | `1536` | Vector dimension (match your embedding model!) |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | `localhost/5433/docgraph/...` | Relational + vector DB |
| `NEO4J_URI/USER/PASSWORD` | `bolt://localhost:7687/neo4j/...` | Graph DB |
| `CHUNK_SIZE / CHUNK_OVERLAP` | `1200 / 200` | Chunking parameters |
| `MAX_AGENT_ITERATIONS` | `3` | Max plan→retrieve→grade loops |

## LLM backbone: any OpenAI-compatible endpoint

The backbone does **not** have to be an OpenAI model. Any endpoint speaking the
OpenAI chat-completions API works — just set `OPENAI_BASE_URL` and `OPENAI_MODEL`:

| Backend | `OPENAI_BASE_URL` | Example `OPENAI_MODEL` |
|---------|-------------------|------------------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| vLLM (self-hosted) | `http://localhost:8000/v1` | `meta-llama/Llama-3.1-70B-Instruct` |
| Ollama | `http://localhost:11434/v1` | `llama3.1` |
| LiteLLM proxy | `http://localhost:4000/v1` | any routed model name |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4` |
| Azure OpenAI (compat) | `https://<resource>.openai.azure.com/openai/v1` | deployed model name |

Notes:
- If the endpoint doesn't support JSON mode (`response_format`), the agent automatically
  falls back to plain calls with JSON extraction — no configuration needed.
- Embeddings must be served by the **same endpoint** (`/embeddings` API). When using e.g.
  vLLM or Ollama, set `OPENAI_EMBEDDING_MODEL` and `EMBEDDING_DIM` to match (e.g.
  `nomic-embed-text` with `768`). Keep the embedding model consistent across ingests —
  vectors from different models are not comparable.

## Project layout

```
app/
├── main.py               # FastAPI service (/ingest, /query, /health)
├── config.py             # Pydantic settings (env-driven)
├── llm.py                # OpenAI-compatible client factory + JSON-mode fallback
├── ingestion/
│   ├── parser.py         # Docling: PDF/DOCX/XLSX → structured documents
│   ├── chunker.py        # Token-aware chunking
│   ├── embedder.py       # OpenAI embeddings
│   ├── entity_extract.py # LLM entity + relation extraction (graph linking)
│   └── pipeline.py       # Orchestrates the full ingestion flow
├── db/
│   ├── postgres.py       # Relational schema + pgvector store
│   └── neo4j_client.py   # Graph store (cross-document linking)
└── agent/
    ├── state.py          # LangGraph agent state
    ├── tools.py          # vector_search / sql_query / graph_query tools
    ├── graph.py          # The corrective-RAG LangGraph agent
    └── prompts.py        # Prompt templates
scripts/
├── ingest_folder.py      # CLI: ingest a folder recursively
└── ask.py                # CLI: one-shot agent query
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness + DB connectivity |
| `/ingest/file` | POST | Upload & ingest one document (multipart) |
| `/ingest/folder` | POST | Ingest a server-side folder `{"path": "..."}` |
| `/query` | POST | Ask the agent `{"question": "..."}` |
| `/documents` | GET | List ingested documents |

## How cross-document linking works

1. Docling extracts text, tables, and structure from each document.
2. An LLM extracts **entities** and **relations** from each chunk (structured JSON output).
3. Entities are canonicalized by name/type and stored in Neo4j.
4. `(:Chunk)-[:MENTIONS]->(:Entity)` and `(:Entity)-[:RELATED {evidence}]->(:Entity)` edges
   create the graph. When the same entity appears in multiple documents, the graph naturally
   links those documents — enabling queries like *"which documents discuss the same suppliers?"*
   via multi-hop traversal.

## Development

```bash
make test        # pytest (requires services up for integration tests)
make lint        # ruff
docker compose down -v   # tear everything down
```

## Security note

The agent's SQL and Cypher tools are **read-only** (SELECT/MATCH only, enforced with
read-only transactions and query parsing). Never expose the service publicly without
adding your own authentication layer.
