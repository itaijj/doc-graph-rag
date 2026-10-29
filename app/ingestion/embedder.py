"""Local embeddings for chunk text using a Hugging Face model (default: BAAI/bge-m3)."""

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        s = get_settings()
        self.model_name = s.embedding_model
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_embedding_dimension()
        if dim != s.embedding_dim:
            raise RuntimeError(
                f"{self.model_name} produces {dim}-dim vectors but EMBEDDING_DIM={s.embedding_dim}."
            )

    def embed(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed a list of texts in batches (L2-normalized, suited to cosine search)."""
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        logger.info("Embedded %d texts with %s", len(texts), self.model_name)
        return vectors.tolist()
