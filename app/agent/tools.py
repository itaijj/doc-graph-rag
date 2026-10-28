"""Retriever tools the agent uses: vector search, read-only SQL, read-only Cypher."""

import logging
from typing import Any

from app.db.neo4j_client import Neo4jStore
from app.db.postgres import PostgresStore
from app.ingestion.embedder import Embedder
from app.config import get_settings

logger = logging.getLogger(__name__)


class RetrieverToolkit:
    """Bundles the three read-only retrievers the agent can invoke."""

    def __init__(self) -> None:
        self.pg = PostgresStore()
        self.neo = Neo4jStore()
        self.embedder = Embedder()
        self.settings = get_settings()

    def vector_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Semantic search over document chunks."""
        k = top_k or self.settings.vector_search_top_k
        embedding = self.embedder.embed([query])[0]
        return self.pg.vector_search(embedding, top_k=k)

    def sql_query(self, query: str) -> list[dict[str, Any]]:
        """Read-only SELECT against the relational metadata tables."""
        return self.pg.read_only_sql(query)

    def graph_query(self, query: str) -> list[dict[str, Any]]:
        """Read-only Cypher against the entity/relation graph."""
        return self.neo.read_only_cypher(query)

    def execute(self, plan_entry: dict[str, Any]) -> dict[str, Any]:
        """Execute one planned retriever call; errors are captured as evidence."""
        tool = plan_entry.get("tool")
        query = plan_entry.get("query", "")
        try:
            if tool == "vector":
                payload = self.vector_search(query)
            elif tool == "sql":
                payload = self.sql_query(query)
            elif tool == "graph":
                payload = self.graph_query(query)
            else:
                return {"source": str(tool), "error": f"unknown tool '{tool}'"}
        except Exception as exc:
            logger.warning("Retriever %s failed: %s", tool, exc)
            return {"source": str(tool), "query": query, "error": str(exc)}
        return {"source": str(tool), "query": query, "payload": payload}

    def close(self) -> None:
        self.pg.close()
        self.neo.close()
