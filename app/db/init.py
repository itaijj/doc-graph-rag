"""CLI entry point: initialize Postgres + Neo4j schemas."""

import logging

from app.db.neo4j_client import Neo4jStore
from app.db.postgres import PostgresStore

logging.basicConfig(level=logging.INFO)


def main() -> None:
    pg = PostgresStore()
    pg.init_schema()
    pg.close()

    neo = Neo4jStore()
    neo.init_schema()
    neo.close()
    print("✅ Schemas initialized (PostgreSQL + pgvector, Neo4j)")


if __name__ == "__main__":
    main()
