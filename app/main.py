"""FastAPI service: ingestion endpoints + agent query endpoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.graph import DocGraphAgent
from app.config import get_settings
from app.db.neo4j_client import Neo4jStore
from app.db.postgres import PostgresStore
from app.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

_pipeline: Optional[IngestionPipeline] = None
_agent: Optional[DocGraphAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _agent
    _pipeline = IngestionPipeline()
    _agent = DocGraphAgent(toolkit=_agent_toolkit(_pipeline))
    yield
    if _pipeline:
        _pipeline.close()


def _agent_toolkit(pipeline: IngestionPipeline):
    """Reuse the pipeline's connections for the agent's retrievers."""
    from app.agent.tools import RetrieverToolkit

    kit = RetrieverToolkit.__new__(RetrieverToolkit)
    kit.pg = pipeline.pg
    kit.neo = pipeline.neo
    kit.embedder = pipeline.embedder
    kit.settings = get_settings()
    return kit


app = FastAPI(title="Doc-Graph-RAG", version="0.1.0", lifespan=lifespan)


class FolderRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    ok_pg = ok_neo = False
    try:
        pg = PostgresStore()
        pg.read_only_sql("SELECT 1 AS ok")
        pg.close()
        ok_pg = True
    except Exception:
        pass
    try:
        neo = Neo4jStore()
        neo.read_only_cypher("MATCH (n) RETURN count(n) AS c LIMIT 1")
        neo.close()
        ok_neo = True
    except Exception:
        pass
    return {
        "status": "ok" if (ok_pg and ok_neo) else "degraded",
        "postgres": ok_pg,
        "neo4j": ok_neo,
        "llm_base_url": settings.openai_base_url,
        "model": settings.openai_model,
    }


@app.post("/ingest/file")
async def ingest_file(file: UploadFile) -> dict:
    data = await file.read()
    from app.ingestion.parser import parse_bytes

    parsed = parse_bytes(file.filename or "upload", data)
    import hashlib

    return _pipeline._ingest_parsed(
        parsed, source_path=file.filename or "upload",
        checksum=hashlib.sha256(data).hexdigest(),
    )


@app.post("/ingest/folder")
def ingest_folder(req: FolderRequest) -> dict:
    folder = Path(req.path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"'{req.path}' is not a directory")
    results = _pipeline.ingest_folder(folder)
    return {
        "total": len(results),
        "ingested": sum(1 for r in results if r.get("status") == "ingested"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }


@app.get("/documents")
def list_documents() -> list[dict]:
    return _pipeline.pg.list_documents()


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return _agent.run(req.question)
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
