"""Application configuration — all settings are environment-driven (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM (any OpenAI-compatible endpoint) ──────────────────
    # openai_base_url may point at OpenAI, Azure OpenAI (compat mode), vLLM,
    # Ollama (http://localhost:11434/v1), LiteLLM, OpenRouter, ...
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    # Embeddings run locally via sentence-transformers (Hugging Face model id).
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    openai_timeout: float = 120.0

    # ── PostgreSQL (relational + pgvector) ──────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "docgraph"
    postgres_user: str = "docgraph"
    postgres_password: str = "docgraph"

    # ── Neo4j (graph) ───────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphpassword"

    # ── Ingestion ───────────────────────────────────────────────
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_entities_per_chunk: int = 15

    # ── Agent ───────────────────────────────────────────────────
    max_agent_iterations: int = 3
    vector_search_top_k: int = 6

    # ── API ─────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def postgres_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
