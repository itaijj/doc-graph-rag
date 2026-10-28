"""End-to-end ingestion pipeline:

    file → Docling parse → chunk → embed → PostgreSQL/pgvector
                                    └→ LLM entity extraction → Neo4j graph
"""

import hashlib
import logging
from pathlib import Path

from app.db.neo4j_client import Neo4jStore
from app.db.postgres import PostgresStore
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import Embedder
from app.ingestion.entity_extract import EntityExtractor
from app.ingestion.parser import ParsedDocument, parse_file

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        self.pg = PostgresStore()
        self.neo = Neo4jStore()
        self.embedder = Embedder()
        self.extractor = EntityExtractor()

    def ingest_file(self, path: Path) -> dict:
        """Ingest a single file. Returns a summary dict."""
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        if self.pg.document_exists(checksum):
            logger.info("Skipping %s (already ingested, checksum match)", path.name)
            return {"status": "skipped", "file": str(path), "reason": "duplicate"}

        parsed = parse_file(path)
        return self._ingest_parsed(parsed, source_path=str(path), checksum=checksum)

    def _ingest_parsed(
        self, parsed: ParsedDocument, source_path: str, checksum: str
    ) -> dict:
        # 1. Relational: document row
        doc_id = self.pg.insert_document(
            title=parsed.title,
            source_path=source_path,
            doc_type=parsed.doc_type,
            checksum=checksum,
            metadata=parsed.metadata,
        )
        # 2. Graph: document node
        self.neo.insert_document(doc_id, parsed.title, parsed.doc_type)

        # 3. Chunk + embed + extract entities
        chunks = chunk_document(parsed)
        embeddings = self.embedder.embed([c.content for c in chunks])

        num_entities = 0
        num_relations = 0
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = self.pg.insert_chunk(
                document_id=doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                page_numbers=chunk.page_numbers,
                metadata={"source_name": parsed.metadata.get("source_name")},
                embedding=embedding,
            )
            self.neo.insert_chunk(chunk_id, doc_id, chunk.chunk_index)

            extraction = self.extractor.extract(parsed.title, chunk.content)
            self.neo.insert_entities_and_relations(
                chunk_id, extraction["entities"], extraction["relations"]
            )
            num_entities += len(extraction["entities"])
            num_relations += len(extraction["relations"])

        self.pg.finalize_document(doc_id, num_chunks=len(chunks))

        summary = {
            "status": "ingested",
            "file": source_path,
            "document_id": str(doc_id),
            "num_chunks": len(chunks),
            "num_entities": num_entities,
            "num_relations": num_relations,
        }
        logger.info("Ingested %s: %s", parsed.title, summary)
        return summary

    def ingest_folder(self, folder: Path) -> list[dict]:
        """Recursively ingest all supported files in a folder."""
        from app.ingestion.parser import SUPPORTED_EXTENSIONS

        results = []
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    results.append(self.ingest_file(path))
                except Exception:
                    logger.exception("Failed to ingest %s", path)
                    results.append({"status": "error", "file": str(path)})
        return results

    def close(self) -> None:
        self.pg.close()
        self.neo.close()
