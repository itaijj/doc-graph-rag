"""OpenAI-compatible embeddings for chunk text (endpoint is configurable)."""

import logging

from app.llm import get_openai_client
from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — embeddings require it.")
        self.client = get_openai_client()
        self.model = s.openai_embedding_model

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of texts in batches."""
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:8000] for t in texts[i : i + batch_size]]  # guard token limits
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        logger.info("Embedded %d texts with %s", len(texts), self.model)
        return out
