"""Smoke tests.

Unit tests (no services needed): chunker safety, SQL/Cypher read-only guards.
Integration tests (need `docker compose up -d` + OPENAI_API_KEY): marked with
`pytest.mark.integration` and skipped automatically when services are unavailable.
"""

import pytest

from app.config import get_settings
from app.ingestion.chunker import _hard_split
from app.ingestion.parser import ParsedDocument, SUPPORTED_EXTENSIONS, parse_file  # noqa: F401
import tiktoken


# ── Unit ─────────────────────────────────────────────────────────────


def test_supported_extensions():
    assert {".pdf", ".docx", ".xlsx"} <= SUPPORTED_EXTENSIONS


def test_hard_split_respects_budget():
    enc = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"word{i}" for i in range(2000))
    parts = _hard_split(text, enc, 200)
    assert len(parts) > 1
    for p in parts:
        assert len(enc.encode(p)) <= 200


def test_sql_guard_rejects_writes():
    from app.db.postgres import PostgresStore

    store = PostgresStore.__new__(PostgresStore)  # no connection needed for guard logic
    with pytest.raises(ValueError):
        store.read_only_sql("DELETE FROM documents")
    with pytest.raises(ValueError):
        store.read_only_sql("SELECT 1; DROP TABLE documents")
    store.pool = None  # avoid close() side effects


def test_cypher_guard_rejects_writes():
    from app.db.neo4j_client import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)
    with pytest.raises(ValueError):
        store.read_only_cypher("MATCH (n) DELETE n")
    with pytest.raises(ValueError):
        store.read_only_cypher("CREATE (n:X)")
    store.driver = None


# ── Integration ──────────────────────────────────────────────────────

pytestmark_integration = pytest.mark.integration


def _services_up() -> bool:
    try:
        from app.db.neo4j_client import Neo4jStore
        from app.db.postgres import PostgresStore

        pg = PostgresStore()
        pg.read_only_sql("SELECT 1 AS ok")
        pg.close()
        neo = Neo4jStore()
        neo.read_only_cypher("MATCH (n) RETURN count(n) AS c LIMIT 1")
        neo.close()
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _services_up(), reason="databases not running")
def test_postgres_roundtrip():
    import uuid

    from app.db.postgres import PostgresStore

    s = get_settings()
    store = PostgresStore()
    store.init_schema()
    doc_id = store.insert_document(
        "t", "t.pdf", "pdf", f"test-{uuid.uuid4()}", {}
    )
    store.insert_chunk(doc_id, 0, "hello world", 2, [1], {}, [0.1] * s.embedding_dim)
    store.finalize_document(doc_id, 1)
    hits = store.vector_search([0.1] * s.embedding_dim, top_k=5)
    assert any(h["content"] == "hello world" for h in hits)
    # cleanup (write via raw pool, bypassing the read-only guard intentionally)
    with store.pool.connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    store.close()
