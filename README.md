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

## Schema-driven field extraction (optional)

During ingestion the app can extract a fixed set of fields from each **PDF** according to a
[JSON Schema](https://json-schema.org/) you provide, and save the result as one JSON file per
document — ready to be aggregated later into statistics. Fields may be scalars, enums, nested
objects, or **lists of JSON objects** (e.g. invoice line items).

It uses Docling's experimental `DocumentExtractor` with the local
[`numind/NuExtract-2.0-2B`](https://huggingface.co/numind/NuExtract-2.0-2B) vision-language model.

### Quick use

```bash
# 1. install (qwen-vl-utils and jsonschema are included)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. point the app at a schema (an example is provided)
echo 'EXTRACTION_SCHEMA_PATH=./schemas/example_extraction_schema.json' >> .env

# 3. ingest as usual — each new PDF also produces ./extractions/<title>.json
make ingest FOLDER=./my_docs
```

### Schema example

```json
{
  "type": "object",
  "properties": {
    "document_title": {"type": "string"},
    "document_type": {"type": "string", "enum": ["invoice", "contract", "report", "other"]},
    "total_amount": {"type": "number"},
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "quantity": {"type": "integer"},
          "price": {"type": "number"}
        }
      }
    }
  }
}
```

Supported: `string`, `integer`, `number`, `boolean`, `enum`, `format: date`, nested objects, arrays of
scalars / objects, and local `$ref`/`$defs`. Unsupported constructs (`oneOf`, `anyOf`, `allOf`, …)
raise a clear error at startup.

### Output

One file per document in `EXTRACTION_OUTPUT_DIR` (default `./extractions`, git-ignored), named by the
document title (a short checksum suffix is added only if two different files share a title):

```json
{
  "source_name": "invoice.pdf", "checksum": "…", "document_id": "…", "schema_hash": "…",
  "status": "extracted",
  "data":  { "document_title": "…", "line_items": [ {"name": "…", "quantity": 2, "price": 10.0} ] },
  "pages": [ { "page_no": 1, "extracted_data": {…}, "raw_text": "…", "errors": [] } ],
  "errors": []
}
```

- The model runs **once per page**; `data` merges the pages (lists are concatenated, scalars take the
  first non-null value) and `pages` keeps the raw per-page output for traceability.
- `status` is `extracted`, `partial`, `extraction_skipped` (DOCX/XLSX are not supported by the
  extractor) or `extraction_failed`. Extraction problems never abort ingestion.
- Read all results later with `app.ingestion.extraction_store.iter_extractions()`.

### Things to know

- The model (~4.2 GB) is downloaded from Hugging Face on first use.
- **GPU strongly recommended.** On CPU a 2-page invoice takes roughly 10–15 minutes.
- Small models make mistakes — always spot-check results against the source documents.
- The ingestion summary includes `extraction_status` (`disabled` when no schema is configured).

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL of **any OpenAI-compatible endpoint** (see below) |
| `OPENAI_API_KEY` | — | **Required** for embeddings + LLM (any dummy value for local servers) |
| `OPENAI_MODEL` | `gpt-4o` | Backbone LLM model name — **any model your endpoint serves** |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Local Hugging Face embedding model (sentence-transformers; downloaded on first use) |
| `EMBEDDING_DIM` | `1024` | Vector dimension (match your embedding model!) |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | `localhost/5433/docgraph/...` | Relational + vector DB |
| `NEO4J_URI/USER/PASSWORD` | `bolt://localhost:7687/neo4j/...` | Graph DB |
| `CHUNK_SIZE / CHUNK_OVERLAP` | `1200 / 200` | Chunking parameters |
| `EXTRACTION_SCHEMA_PATH` | *(empty — disabled)* | Path to a JSON Schema file of fields to extract from each PDF during ingestion |
| `EXTRACTION_OUTPUT_DIR` | `./extractions` | Directory where one `<title>.json` extraction result per document is written |
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
- Embeddings run **locally** with a Hugging Face model via `sentence-transformers` (default
  `BAAI/bge-m3`, 1024 dims) — no embeddings API is needed. To switch models, set
  `EMBEDDING_MODEL` and `EMBEDDING_DIM` to match. Keep the embedding model consistent across
  ingests — vectors from different models are not comparable.

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
│   ├── field_extract.py  # JSON Schema → NuExtract template, per-page extraction + merge
│   ├── extraction_store.py # One JSON file per document (extraction results)
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
