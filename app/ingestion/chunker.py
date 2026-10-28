"""Token-aware chunking of parsed documents with overlap."""

import logging
from dataclasses import dataclass

import tiktoken

from app.config import get_settings
from app.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_index: int
    content: str
    token_count: int
    page_numbers: list[int]


def _num_tokens(text: str, encoding) -> int:
    return len(encoding.encode(text))


def chunk_document(parsed: ParsedDocument) -> list[Chunk]:
    """Split a parsed document into overlapping, token-bounded chunks.

    Small blocks (e.g. table rows) are packed together; large blocks are split.
    """
    s = get_settings()
    enc = tiktoken.get_encoding("cl100k_base")
    max_tokens, overlap = s.chunk_size, s.chunk_overlap

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    buffer_pages: set[int] = set()

    def flush():
        nonlocal buffer, buffer_tokens, buffer_pages
        if not buffer:
            return
        text = "\n\n".join(buffer)
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                content=text,
                token_count=buffer_tokens,
                page_numbers=sorted(buffer_pages) or [1],
            )
        )
        # Keep trailing overlap.
        tail: list[str] = []
        tail_tokens = 0
        for part in reversed(buffer):
            t = _num_tokens(part, enc)
            if tail_tokens + t > overlap and tail:
                break
            tail.insert(0, part)
            tail_tokens += t
        buffer = list(tail)
        buffer_tokens = tail_tokens
        buffer_pages = set(buffer_pages)  # pages carry over with the overlap

    for text, pages in zip(parsed.chunks_raw, parsed.page_map):
        tokens = _num_tokens(text, enc)
        if tokens > max_tokens:
            # Split an oversized block by sentences-ish (paragraphs, then hard split).
            flush()
            for part in _hard_split(text, enc, max_tokens):
                pt = _num_tokens(part, enc)
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=part,
                        token_count=pt,
                        page_numbers=pages or [1],
                    )
                )
            continue
        if buffer_tokens + tokens > max_tokens:
            flush()
        buffer.append(text)
        buffer_tokens += tokens
        buffer_pages.update(pages or [1])

    flush()
    logger.info("Chunked '%s' into %d chunks", parsed.title, len(chunks))
    return chunks


def _hard_split(text: str, enc, max_tokens: int) -> list[str]:
    """Hard-split an oversized block into <=max_tokens pieces."""
    parts: list[str] = []
    remaining = text
    while _num_tokens(remaining, enc) > max_tokens:
        # Approximate character position for the token budget, then snap to whitespace.
        ratio = max_tokens / _num_tokens(remaining, enc)
        cut = max(int(len(remaining) * ratio), 1)
        while cut < len(remaining) and not remaining[cut].isspace():
            cut += 1
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts
