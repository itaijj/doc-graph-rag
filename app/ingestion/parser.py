"""Docling-based parser for PDF, Word (.docx), and Excel (.xlsx) documents."""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}


@dataclass
class ParsedDocument:
    """Normalized representation of a parsed document, independent of source format."""
    title: str
    doc_type: str                              # pdf | docx | xlsx
    chunks_raw: list[str] = field(default_factory=list)     # ordered text blocks
    page_map: list[list[int]] = field(default_factory=list)  # pages per block
    metadata: dict[str, Any] = field(default_factory=dict)


def _iter_docling_items(doc):
    """Yield (text, page_numbers) pairs from a DoclingDocument."""
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            # Fall back to table export (Docling represents tables as TableItem).
            if hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown()
            if not text or not text.strip():
                continue
        pages: list[int] = []
        prov = getattr(item, "prov", None) or []
        for p in prov:
            page_no = getattr(p, "page_no", None)
            if page_no is not None:
                pages.append(page_no)
        yield text.strip(), pages


def parse_file(path: Path) -> ParsedDocument:
    """Parse a single PDF/DOCX/XLSX file with Docling into a normalized document."""
    from docling.document_converter import DocumentConverter

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    converter = DocumentConverter()
    logger.info("Parsing %s ...", path.name)
    result = converter.convert(str(path))
    dl_doc = result.document

    parsed = ParsedDocument(title=path.stem, doc_type=ext.lstrip("."))
    for text, pages in _iter_docling_items(dl_doc):
        parsed.chunks_raw.append(text)
        parsed.page_map.append(pages or [1])

    parsed.metadata = {
        "source_name": path.name,
        "num_blocks": len(parsed.chunks_raw),
        "num_pages": len(result.pages) if hasattr(result, "pages") else None,
    }
    logger.info("Parsed %s: %d text blocks", path.name, len(parsed.chunks_raw))
    return parsed


def parse_bytes(filename: str, data: bytes) -> ParsedDocument:
    """Parse an in-memory uploaded file by spooling it to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        parsed = parse_file(tmp_path)
        parsed.title = Path(filename).stem
        parsed.metadata["source_name"] = filename
        return parsed
    finally:
        tmp_path.unlink(missing_ok=True)
