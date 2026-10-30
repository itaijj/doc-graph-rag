"""JSON-file persistence for schema-driven field extraction results.

One human-readable ``<title>.json`` file is written per document into the extraction output directory
(``settings.extraction_output_dir`` unless an explicit directory is passed).  A later job can load the whole
directory with :func:`iter_extractions` and aggregate the records.

File naming
-----------
The file stem is the sanitized document title (see :func:`sanitize_title`).  If ``<title>.json`` already
exists but stores a *different* checksum (a different file with the same stem), the record goes to
``<title>__<first 8 chars of checksum>.json`` instead.  Reading follows the same two-step order.

Record keys
-----------
``source_name``, ``source_path``, ``checksum``, ``document_id`` (``None`` when not ingested), ``schema_hash``,
``status``, ``extracted_at`` (UTC ISO-8601), ``data``, ``pages``, ``errors``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_SUFFIX = ".json"
_CHECKSUM_PREFIX_LEN = 8
_DEFAULT_STEM = "untitled"
# Leave room for "__<8 chars>.json" inside the common 255-byte filename limit.
_MAX_STEM_BYTES = 200
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Statuses whose record never carries extracted data.
_NO_DATA_STATUSES = frozenset({"extraction_skipped", "extraction_failed"})


def sanitize_title(title: str | None) -> str:
    """Return a filesystem-safe file stem for ``title`` (never empty, never a path)."""
    stem = unicodedata.normalize("NFC", title or "")
    stem = _UNSAFE_CHARS.sub("_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if stem.split(".")[0].upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    # Truncate by UTF-8 bytes without splitting a character.
    stem = stem.encode("utf-8")[:_MAX_STEM_BYTES].decode("utf-8", errors="ignore").strip(" .")
    return stem or _DEFAULT_STEM


def _resolve_dir(output_dir: str | Path | None) -> Path:
    """Explicit directory wins; otherwise read the setting lazily (it is a plain string)."""
    return Path(output_dir if output_dir is not None else get_settings().extraction_output_dir)


def _primary_path(title: str, directory: Path) -> Path:
    return directory / f"{sanitize_title(title)}{_SUFFIX}"


def _suffixed_path(title: str, checksum: str, directory: Path) -> Path:
    stem = sanitize_title(title)
    return directory / f"{stem}__{checksum[:_CHECKSUM_PREFIX_LEN]}{_SUFFIX}"


def _read_record(path: Path) -> dict[str, Any] | None:
    """Load a record, returning ``None`` if the file is missing, unreadable or not a JSON object."""
    try:
        with path.open(encoding="utf-8") as fh:
            record = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Could not read extraction file %s: %s", path, exc)
        return None
    return record if isinstance(record, dict) else None


def _find_record_path(title: str, checksum: str, directory: Path) -> Path | None:
    """Existing file for ``(title, checksum)``: ``<title>.json`` first, then the suffixed name."""
    for path in (_primary_path(title, directory), _suffixed_path(title, checksum, directory)):
        record = _read_record(path)
        if record is not None and record.get("checksum") == checksum:
            return path
    return None


def _target_path(title: str, checksum: str, directory: Path) -> Path:
    """Where to write the record for ``(title, checksum)``.

    ``<title>.json`` is used when it is absent, unreadable/corrupt (nothing worth preserving) or already
    holds this checksum; otherwise the checksum-suffixed name is used.
    """
    primary = _primary_path(title, directory)
    record = _read_record(primary)
    if record is None or record.get("checksum") == checksum:
        return primary
    return _suffixed_path(title, checksum, directory)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` via a temp file in the same directory + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_extraction(
    title: str,
    source_path: str | Path,
    checksum: str,
    result: dict[str, Any],
    *,
    source_name: str | None = None,
    document_id: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Persist one extraction ``result`` (the dict returned by ``FieldExtractor.run``) and return its path.

    ``source_name`` defaults to the basename of ``source_path``; pass it explicitly for uploads, where
    ``source_path`` is a temp file.  ``extraction_skipped`` / ``extraction_failed`` records are saved too,
    always with an empty ``data`` dict.
    """
    status = result.get("status", "extraction_failed")
    data = {} if status in _NO_DATA_STATUSES else (result.get("data") or {})
    record: dict[str, Any] = {
        "source_name": source_name if source_name is not None else Path(source_path).name,
        "source_path": str(source_path),
        "checksum": checksum,
        "document_id": document_id,
        "schema_hash": result.get("schema_hash"),
        "status": status,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "pages": result.get("pages") or [],
        "errors": list(result.get("errors") or []),
    }
    directory = _resolve_dir(output_dir)
    path = _target_path(title, checksum, directory)
    _atomic_write_json(path, record)
    return path


def load_extraction(
    title: str, checksum: str, output_dir: str | Path | None = None
) -> dict[str, Any] | None:
    """Return the stored record for ``(title, checksum)``, or ``None`` if there is none."""
    directory = _resolve_dir(output_dir)
    path = _find_record_path(title, checksum, directory)
    return _read_record(path) if path is not None else None


def iter_extractions(output_dir: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield every stored record (sorted by file name); unreadable files are logged and skipped."""
    directory = _resolve_dir(output_dir)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob(f"*{_SUFFIX}")):
        record = _read_record(path)
        if record is not None:
            yield record


def extraction_is_current(
    title: str,
    checksum: str,
    schema_hash: str,
    output_dir: str | Path | None = None,
) -> bool:
    """True if a record exists for this file content *and* schema, so re-extraction can be skipped.

    Requires the stored ``checksum`` and ``schema_hash`` to both match.  Status handling: ``extracted``,
    ``partial`` and ``extraction_skipped`` count as current (skipped is deterministic for a given file, so
    retrying would only repeat it), while ``extraction_failed`` does NOT — failures are often transient
    (model download, out-of-memory), so a re-run should retry them.
    """
    record = load_extraction(title, checksum, output_dir)
    if record is None:
        return False
    return record.get("schema_hash") == schema_hash and record.get("status") != "extraction_failed"
