"""End-to-end ingestion pipeline:

    file → Docling parse → chunk → embed → PostgreSQL/pgvector
                                    └→ LLM entity extraction → Neo4j graph
"""

import hashlib
import logging
import tempfile
from pathlib import Path

from app.config import get_settings
from app.db.neo4j_client import Neo4jStore
from app.db.postgres import PostgresStore
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import Embedder
from app.ingestion.entity_extract import EntityExtractor
from app.ingestion.extraction_store import save_extraction
from app.ingestion.field_extract import FieldExtractor, load_schema
from app.ingestion.parser import ParsedDocument, parse_file

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        # Optional schema-driven field extraction. Enabled only when a schema path is
        # configured. A configured-but-broken schema fails fast (SchemaConversionError /
        # ValueError / OSError propagate) so a misconfiguration is not silently ignored.
        # Built first so a bad schema does not leave DB connections open.
        schema_path = get_settings().extraction_schema_path.strip()
        self.field_extractor: FieldExtractor | None = (
            FieldExtractor(load_schema(schema_path)) if schema_path else None
        )

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
        return self._ingest_parsed(
            parsed, source_path=str(path), checksum=checksum, extract_path=path
        )

    def ingest_bytes(self, filename: str, data: bytes) -> dict:
        """Ingest an uploaded file. The bytes are spooled to a temp file (original
        extension preserved) that stays alive until field extraction has finished."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / f"upload{Path(filename).suffix}"
            tmp_path.write_bytes(data)
            parsed = parse_file(tmp_path)
            parsed.title = Path(filename).stem
            parsed.metadata["source_name"] = filename
            return self._ingest_parsed(
                parsed,
                source_path=filename,
                checksum=hashlib.sha256(data).hexdigest(),
                extract_path=tmp_path,
            )

    def _extract_fields(
        self,
        parsed: ParsedDocument,
        doc_id,
        source_path: str,
        checksum: str,
        extract_path: Path | None,
    ) -> str:
        """Run schema field extraction and save the record. Never raises; returns the
        extraction status (``disabled`` when extraction is off)."""
        if self.field_extractor is None or extract_path is None:
            return "disabled"
        source_name = parsed.metadata.get("source_name")
        try:
            result = self.field_extractor.run(extract_path)
        except Exception as exc:
            logger.exception("Field extraction failed for %s", parsed.title)
            result = {
                "status": "extraction_failed",
                "data": {},
                "pages": [],
                "errors": [f"{type(exc).__name__}: {exc}"],
                "schema_hash": self.field_extractor._schema_hash,
            }
        try:
            save_extraction(
                parsed.title,
                source_path,
                checksum,
                result,
                source_name=source_name,
                document_id=str(doc_id),
            )
        except Exception:
            logger.exception("Could not save extraction record for %s", parsed.title)
            return "extraction_failed"
        return result["status"]

    def _ingest_parsed(
        self,
        parsed: ParsedDocument,
        source_path: str,
        checksum: str,
        extract_path: Path | None = None,
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

        # 4. Schema-driven field extraction (never aborts ingestion)
        extraction_status = self._extract_fields(
            parsed, doc_id, source_path, checksum, extract_path
        )

        summary = {
            "status": "ingested",
            "file": source_path,
            "document_id": str(doc_id),
            "num_chunks": len(chunks),
            "num_entities": num_entities,
            "num_relations": num_relations,
            "extraction_status": extraction_status,
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
