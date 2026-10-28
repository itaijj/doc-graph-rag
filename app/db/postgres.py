"""PostgreSQL store: relational document/chunk metadata + pgvector embeddings."""

import json
import logging
import uuid
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           UUID PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    doc_type     TEXT NOT NULL,          -- pdf | docx | xlsx
    checksum     TEXT NOT NULL,
    num_chunks   INTEGER NOT NULL DEFAULT 0,
    metadata     JSONB  NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    token_count  INTEGER NOT NULL DEFAULT 0,
    page_numbers INTEGER[],
    metadata     JSONB NOT NULL DEFAULT '{}',
    embedding    vector({dim})
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


class PostgresStore:
    """Relational + vector store backed by PostgreSQL/pgvector."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pool = ConnectionPool(
            conninfo=self.settings.postgres_dsn,
            min_size=1,
            max_size=8,
            open=True,
        )
        # Ensure every pooled connection can serialize `vector` types.
        self.pool.configure(self._configure_conn)

    @staticmethod
    def _configure_conn(conn: psycopg.Connection) -> None:
        register_vector(conn)

    def init_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA_SQL.format(dim=self.settings.embedding_dim))
        logger.info("Postgres schema initialized (vector dim=%d)", self.settings.embedding_dim)

    # ── Write path (ingestion) ────────────────────────────────────

    def insert_document(
        self,
        title: str,
        source_path: str,
        doc_type: str,
        checksum: str,
        metadata: dict[str, Any],
    ) -> uuid.UUID:
        doc_id = uuid.uuid4()
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO documents (id, title, source_path, doc_type, checksum, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (doc_id, title, source_path, doc_type, checksum, json.dumps(metadata)),
            )
        return doc_id

    def insert_chunk(
        self,
        document_id: uuid.UUID,
        chunk_index: int,
        content: str,
        token_count: int,
        page_numbers: list[int],
        metadata: dict[str, Any],
        embedding: list[float],
    ) -> uuid.UUID:
        chunk_id = uuid.uuid4()
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO chunks
                       (id, document_id, chunk_index, content, token_count,
                        page_numbers, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    chunk_id,
                    document_id,
                    chunk_index,
                    content,
                    token_count,
                    page_numbers,
                    json.dumps(metadata),
                    embedding,
                ),
            )
        return chunk_id

    def finalize_document(self, document_id: uuid.UUID, num_chunks: int) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET num_chunks = %s WHERE id = %s",
                (num_chunks, document_id),
            )

    def document_exists(self, checksum: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE checksum = %s LIMIT 1", (checksum,)
            ).fetchone()
        return row is not None

    # ── Read path (agent tools) ───────────────────────────────────

    def vector_search(self, query_embedding: list[float], top_k: int = 6) -> list[dict[str, Any]]:
        """Cosine similarity search over chunk embeddings."""
        sql = """
            SELECT c.id,
                   d.title,
                   d.source_path,
                   c.chunk_index,
                   c.content,
                   c.page_numbers,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        with self.pool.connection() as conn:
            register_vector(conn)
            rows = conn.execute(sql, (query_embedding, query_embedding, top_k)).fetchall()
        cols = ["id", "title", "source_path", "chunk_index", "content", "page_numbers", "similarity"]
        return [dict(zip(cols, r)) for r in rows]

    def read_only_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute a strictly read-only SELECT query. Returns list of row dicts."""
        stripped = query.strip().rstrip(";").strip()
        if not stripped.lower().startswith("select") or ";" in stripped:
            raise ValueError("Only a single read-only SELECT statement is allowed.")
        with self.pool.connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            with conn.cursor() as cur:
                cur.execute(stripped)
                cols = [d.name for d in cur.description] if cur.description else []
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    def list_documents(self) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, title, source_path, doc_type, num_chunks, created_at
                   FROM documents ORDER BY created_at DESC"""
            ).fetchall()
        cols = ["id", "title", "source_path", "doc_type", "num_chunks", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def close(self) -> None:
        self.pool.close()
