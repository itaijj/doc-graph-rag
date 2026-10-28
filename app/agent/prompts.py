"""Prompt templates for the LangGraph agent."""

PLAN_PROMPT = """You are the planner of a document-intelligence agent. You have access to \
three retrievers over a corpus of ingested PDF/Word/Excel documents:

1. "vector" — semantic search over document chunks (best for: meaning, topics, finding \
relevant passages).
2. "sql" — read-only SQL over PostgreSQL metadata tables: documents(id, title, source_path, \
doc_type, num_chunks, created_at), chunks(id, document_id, chunk_index, content, token_count, \
page_numbers, metadata) (best for: counts, listings, filtering, structured facts).
3. "graph" — read-only Cypher over a Neo4j graph with nodes (:Document {{id,title,doc_type}}), \
(:Chunk {{id,document_id,chunk_index}}), (:Entity {{key,name,type}}) and relationships \
(:Document)-[:HAS_CHUNK]->(:Chunk), (:Chunk)-[:MENTIONS]->(:Entity), \
(:Entity)-[:RELATED {{relation_type,evidence}}]->(:Entity) (best for: cross-document links, \
shared entities, multi-hop relations between documents).

Question: {question}

Previous rounds (plan + evidence summary, may be empty):
{history}

Decide which retrievers to invoke this round and with what concrete queries.
Return STRICT JSON:
{{
  "retrievers": [
    {{"tool": "vector", "query": "<natural language search query>"}},
    {{"tool": "sql", "query": "<single SELECT statement>"}},
    {{"tool": "graph", "query": "<single MATCH ... RETURN statement>"}}
  ],
  "notes": "<one-sentence rationale>"
}}

Rules:
- Choose only the retrievers that help; 1–3 entries.
- SQL must be a single read-only SELECT against the tables above.
- Cypher must be a single read-only MATCH/RETURN statement.
- If previous evidence was insufficient, refine the queries rather than repeating them."""

GRADE_PROMPT = """You are evaluating whether retrieved evidence is sufficient to answer a \
question. Be strict: sufficient means the answer can be grounded in the evidence.

Question: {question}

Evidence:
{evidence}

Return STRICT JSON:
{{"grade": "sufficient" | "insufficient", "notes": "<what is missing or why it is enough>"}}

Answer "sufficient" if the evidence supports a grounded answer, even if not exhaustive. \
Answer "insufficient" only if the evidence is empty or clearly off-topic."""

SYNTHESIZE_PROMPT = """You are answering a question about a corpus of ingested documents \
(PDF/Word/Excel). Ground your answer ONLY in the evidence below and cite documents inline \
like [Document Title, chunk N].

Question: {question}

Evidence:
{evidence}

Write a direct, well-structured answer. Use the evidence's document titles for citations. \
If parts of the question cannot be answered from the evidence, say so explicitly."""
