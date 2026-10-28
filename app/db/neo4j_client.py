"""Neo4j graph store: entities, relations, and cross-document links.

Graph model:
    (:Document {id, title, doc_type})
    (:Chunk     {id, document_id, chunk_index})
    (:Entity    {name, type})

    (:Document)-[:HAS_CHUNK]->(:Chunk)
    (:Chunk)-[:MENTIONS]->(:Entity)
    (:Entity)-[:RELATED {relation_type, evidence, source_chunk}]->(:Entity)
    (:Document)-[:SHARES_ENTITY]->(:Document)   (computed at query time)
"""

import logging
import uuid
from typing import Any

from neo4j import GraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
]


class Neo4jStore:
    """Graph store for cross-document entity linking."""

    def __init__(self) -> None:
        s = get_settings()
        self.driver = GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)
        )

    def init_schema(self) -> None:
        with self.driver.session() as session:
            for stmt in CONSTRAINTS:
                session.run(stmt)
        logger.info("Neo4j constraints initialized")

    def close(self) -> None:
        self.driver.close()

    # ── Write path (ingestion) ────────────────────────────────────

    def insert_document(self, doc_id: uuid.UUID, title: str, doc_type: str) -> None:
        with self.driver.session() as session:
            session.run(
                """MERGE (d:Document {id: $doc_id})
                   SET d.title = $title, d.doc_type = $doc_type""",
                doc_id=str(doc_id), title=title, doc_type=doc_type,
            )

    def insert_chunk(self, chunk_id: uuid.UUID, doc_id: uuid.UUID, chunk_index: int) -> None:
        with self.driver.session() as session:
            session.run(
                """MATCH (d:Document {id: $doc_id})
                   MERGE (c:Chunk {id: $chunk_id})
                   SET c.document_id = $doc_id, c.chunk_index = $chunk_index
                   MERGE (d)-[:HAS_CHUNK]->(c)""",
                doc_id=str(doc_id), chunk_id=str(chunk_id), chunk_index=chunk_index,
            )

    def insert_entities_and_relations(
        self,
        chunk_id: uuid.UUID,
        entities: list[dict[str, str]],
        relations: list[dict[str, str]],
    ) -> None:
        """Insert extracted entities/relations for a chunk.

        entities:  [{"name": ..., "type": ...}, ...]
        relations: [{"source": ..., "target": ..., "type": ..., "evidence": ...}, ...]
        Entities are canonicalized by `name (type)` key, so the same entity mentioned
        in different documents/chunks merges into one node — linking the documents.
        """
        with self.driver.session() as session:
            for ent in entities:
                session.run(
                    """MERGE (e:Entity {key: $key})
                       SET e.name = $name, e.type = $type""",
                    key=f"{ent['name']} ({ent['type']})",
                    name=ent["name"],
                    type=ent.get("type", "UNKNOWN"),
                )
                session.run(
                    """MATCH (c:Chunk {id: $chunk_id}), (e:Entity {key: $key})
                       MERGE (c)-[:MENTIONS]->(e)""",
                    chunk_id=str(chunk_id), key=f"{ent['name']} ({ent['type']})",
                )
            for rel in relations:
                src_key = f"{rel['source']['name']} ({rel['source']['type']})"
                dst_key = f"{rel['target']['name']} ({rel['target']['type']})"
                session.run(
                    """MATCH (a:Entity {key: $src}), (b:Entity {key: $dst})
                       MERGE (a)-[r:RELATED {relation_type: $rtype}]->(b)
                       SET r.evidence = $evidence, r.source_chunk = $chunk_id""",
                    src=src_key,
                    dst=dst_key,
                    rtype=rel.get("type", "RELATED_TO"),
                    evidence=rel.get("evidence", ""),
                    chunk_id=str(chunk_id),
                )

    # ── Read path (agent tools) ───────────────────────────────────

    def read_only_cypher(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Execute a strictly read-only MATCH query. Returns list of record dicts."""
        stripped = query.strip().rstrip(";").strip()
        lowered = stripped.lower()
        if ";" in stripped:
            raise ValueError("Only a single statement is allowed.")
        if not (lowered.startswith("match") or lowered.startswith("return")
                or lowered.startswith("with") or lowered.startswith("unwind")):
            raise ValueError("Only read-only MATCH/RETURN queries are allowed.")
        banned = ("create", "merge", "delete", "detach", "set ", "remove", "drop", "load csv",
                  "call {", "foreach")
        for kw in banned:
            if kw in lowered:
                raise ValueError(f"Write keyword '{kw.strip()}' is not allowed.")
        with self.driver.session() as session:
            result = session.run(stripped)
            return [dict(r) for r in result.fetch(limit)]

    def related_documents(self, doc_title: str, limit: int = 10) -> list[dict[str, Any]]:
        """Find documents that share entities with the given document (cross-doc links)."""
        q = """
            MATCH (d1:Document {title: $title})-[:HAS_CHUNK]->()-[:MENTIONS]->(e:Entity)
                  <-[:MENTIONS]-()-[:HAS_CHUNK]-(d2:Document)
            WHERE d1 <> d2
            RETURN d2.title AS related_document,
                   collect(DISTINCT e.name)[..10] AS shared_entities,
                   count(DISTINCT e) AS num_shared
            ORDER BY num_shared DESC
            LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(q, title=doc_title, limit=limit)]
