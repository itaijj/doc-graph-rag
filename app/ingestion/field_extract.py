"""Schema-driven field extraction: JSON Schema -> NuExtract template conversion.

The converter is pure (no model, no I/O besides ``load_schema``) so it can be unit-tested directly.

Template conventions (NuExtract 2.0, ``numind/NuExtract-2.0-*``):
- scalar types are strings: ``"string"``, ``"integer"``, ``"number"``, ``"boolean"``, ``"date-time"``
- a single-choice enum is a list of option strings: ``["a", "b"]``
- a multi-choice enum is a list containing that list: ``[["a", "b"]]``
- a nested object is a nested dict; a list of objects is ``[{...}]``; a list of scalars is ``["string"]``

``FieldExtractor`` wraps ``DocumentExtractor`` from Docling with lazy loading so the module can be
imported without pulling in torch/transformers.  Call ``FieldExtractor.run(path)`` to extract and
merge all pages into one result dict::

    {
        "status": "extracted" | "partial" | "extraction_skipped" | "extraction_failed",
        "data":   {<merged fields per schema>},
        "pages":  [{"page_no": int, "extracted_data": dict|None, "raw_text": str|None, "errors": [...]}],
        "errors": ["..."],
        "schema_hash": "<sha256 hex>",
    }
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Extensions the Docling DocumentExtractor actually supports.
_EXTRACTABLE_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
}

_SCALAR_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
}
_DATE_FORMATS = {"date", "date-time"}
_UNSUPPORTED_KEYWORDS = ("oneOf", "anyOf", "allOf", "not", "if", "patternProperties")


class SchemaConversionError(ValueError):
    """Raised when a JSON Schema is invalid or uses constructs we cannot convert."""


def load_schema(path: str | Path) -> dict[str, Any]:
    """Read a JSON Schema file and return it as a dict."""
    p = Path(path)
    try:
        schema = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise SchemaConversionError(f"Cannot read schema file {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise SchemaConversionError(f"Schema file {p} is not valid JSON: {e}") from e
    if not isinstance(schema, dict):
        raise SchemaConversionError(f"Schema file {p} must contain a JSON object.")
    return schema


def schema_to_template(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema (top-level object with properties) to a NuExtract template dict."""
    if not isinstance(schema, dict):
        raise SchemaConversionError("Schema must be a JSON object.")
    root = _resolve(schema, schema, ())
    if _node_type(root, "<root>") != "object":
        raise SchemaConversionError("Top-level schema must have type 'object'.")
    if not isinstance(root.get("properties"), dict) or not root["properties"]:
        raise SchemaConversionError("Top-level schema must define a non-empty 'properties' object.")
    return _convert_object(root, schema, "<root>", ())


def _resolve(node: Any, root: dict[str, Any], refs: tuple[str, ...]) -> dict[str, Any]:
    """Follow local ``$ref`` pointers (``#/...``) until a concrete schema node is reached."""
    seen = refs
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise SchemaConversionError(f"Only local '$ref' pointers are supported, got {ref!r}.")
        if ref in seen:
            raise SchemaConversionError(f"Recursive '$ref' is not supported: {ref!r}.")
        seen = (*seen, ref)
        target: Any = root
        for part in ref[1:].split("/")[1:] if ref != "#" else []:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                raise SchemaConversionError(f"Unresolvable '$ref': {ref!r}.")
            target = target[part]
        node = target
    if not isinstance(node, dict):
        raise SchemaConversionError("Schema nodes must be JSON objects.")
    for kw in _UNSUPPORTED_KEYWORDS:
        if kw in node:
            raise SchemaConversionError(f"Unsupported schema construct '{kw}'.")
    return node


def _node_type(node: dict[str, Any], name: str) -> str:
    """Return the single JSON type of a node (a ``["x", "null"]`` union is treated as ``"x"``)."""
    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) != 1:
            raise SchemaConversionError(f"Field '{name}': type unions are not supported ({t}).")
        t = non_null[0]
    if t is None:
        if "properties" in node:
            return "object"
        if "items" in node:
            return "array"
        if "enum" in node:
            return "string"
        raise SchemaConversionError(f"Field '{name}': missing 'type'.")
    if not isinstance(t, str):
        raise SchemaConversionError(f"Field '{name}': invalid 'type' {t!r}.")
    return t


def _enum_options(node: dict[str, Any], name: str) -> list[str]:
    options = node["enum"]
    if not isinstance(options, list) or not options or not all(isinstance(o, str) for o in options):
        raise SchemaConversionError(f"Field '{name}': 'enum' must be a non-empty list of strings.")
    return list(options)


def _convert_object(
    node: dict[str, Any], root: dict[str, Any], name: str, refs: tuple[str, ...]
) -> dict[str, Any]:
    props = node.get("properties")
    if not isinstance(props, dict) or not props:
        raise SchemaConversionError(f"Field '{name}': object must define non-empty 'properties'.")
    return {key: _convert_node(sub, root, key, refs) for key, sub in props.items()}


def _convert_node(raw: Any, root: dict[str, Any], name: str, refs: tuple[str, ...]) -> Any:
    node = _resolve(raw, root, refs)
    t = _node_type(node, name)
    if t == "object":
        return _convert_object(node, root, name, refs)
    if t == "array":
        if "items" not in node:
            raise SchemaConversionError(f"Field '{name}': array must define 'items'.")
        item = _resolve(node["items"], root, refs)
        item_type = _node_type(item, name)
        if item_type == "array":
            raise SchemaConversionError(f"Field '{name}': nested arrays are not supported.")
        if item_type == "object":
            return [_convert_object(item, root, name, refs)]
        if "enum" in item:
            return [_enum_options(item, name)]  # multi-choice
        return [_scalar(item, item_type, name)]
    if "enum" in node:
        return _enum_options(node, name)  # single-choice
    return _scalar(node, t, name)


def _scalar(node: dict[str, Any], json_type: str, name: str) -> str:
    if json_type == "string" and node.get("format") in _DATE_FORMATS:
        return "date-time"
    if json_type not in _SCALAR_TYPES:
        raise SchemaConversionError(f"Field '{name}': unsupported type '{json_type}'.")
    return _SCALAR_TYPES[json_type]


# ---------------------------------------------------------------------------
# Helpers: schema field classification (used by merge_pages)
# ---------------------------------------------------------------------------


def _schema_hash(schema: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of the schema content."""
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _is_list_field(key: str, schema: dict[str, Any]) -> bool:
    """Return True when the schema property *key* is an array (list field)."""
    props = schema.get("properties", {})
    if key not in props:
        return False
    node = props[key]
    # Unwrap nullable union
    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if len(non_null) == 1 else None
    if t == "array":
        return True
    # Missing explicit type but has "items"
    return t is None and "items" in node


# ---------------------------------------------------------------------------
# merge_pages: pure, unit-testable
# ---------------------------------------------------------------------------


def merge_pages(
    pages: list[dict[str, Any]],
    template: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Merge per-page extracted dicts into one record.

    Rules (applied recursively for nested objects):
    - **List fields** (array in the schema): items from all pages are concatenated in page order.
      Duplicate elimination is *not* performed (caller responsibility).
    - **Scalar fields** (non-array): first non-null value across pages wins; others are discarded.
    - **Nested objects**: merged recursively with the same rules.
    - Every key present in *template* appears in the result even when no page provided a value
      (``None`` for scalars, ``[]`` for lists) so every record has the same shape.

    :param pages: ``extracted_data`` dicts from each page, in page order.  ``None`` entries are
        silently skipped.
    :param template: NuExtract template dict produced by :func:`schema_to_template`.
    :param schema: the original JSON Schema dict (needed to classify list vs scalar for top-level
        and nested fields).
    :returns: A plain JSON-serialisable dict with the same shape as *template*.
    """
    non_null_pages = [p for p in pages if p is not None]
    return _merge_object(non_null_pages, template, schema)


def _merge_object(
    page_dicts: list[dict[str, Any]],
    template_node: dict[str, Any],
    schema_node: dict[str, Any],
) -> dict[str, Any]:
    """Merge a list of per-page dicts for one object level."""
    result: dict[str, Any] = {}
    for key, tmpl_value in template_node.items():
        page_values = [p[key] for p in page_dicts if isinstance(p, dict) and key in p]
        result[key] = _merge_value(key, page_values, tmpl_value, schema_node)
    return result


def _merge_value(
    key: str,
    page_values: list[Any],
    tmpl_value: Any,
    schema_node: dict[str, Any],
) -> Any:
    """Merge per-page values for a single field.

    ``tmpl_value`` determines the *shape* of the field (dict → nested object, list → array).
    ``schema_node`` is consulted to decide whether a list template value represents a list field
    (array) or a single-choice enum (which is also represented as a list of strings in the
    template but is semantically a scalar).
    """
    if isinstance(tmpl_value, dict):
        # Nested object — recurse; get the sub-schema from properties if available
        props = schema_node.get("properties", {}) if isinstance(schema_node, dict) else {}
        sub_schema = props.get(key, {})
        # Resolve to actual object schema (handle array-of-objects: schema has items)
        if sub_schema.get("type") == "array" or (
            sub_schema.get("type") is None and "items" in sub_schema
        ):
            sub_schema = sub_schema.get("items", {})
        sub_page_dicts = [v for v in page_values if isinstance(v, dict)]
        return _merge_object(sub_page_dicts, tmpl_value, sub_schema)

    if isinstance(tmpl_value, list):
        # Template list can mean three things:
        #   1. list-of-objects  → [{ ... }]           → concat all page lists
        #   2. multi-choice enum → [["a","b"]]         → concat (treat as list)
        #   3. single-choice enum → ["a","b"]          → scalar; use schema to decide
        # We distinguish case 3 by checking the JSON Schema: if the property type is NOT array
        # (and has no "items"), it is a scalar enum field.
        is_list = _is_list_field(key, schema_node)
        if not is_list:
            # Scalar enum: return first non-null value
            for v in page_values:
                if v is not None:
                    return v
            return None

        # True list field: concatenate all non-null list values from pages
        merged: list[Any] = []
        for v in page_values:
            if isinstance(v, list):
                merged.extend(v)
            elif v is not None:
                merged.append(v)
        return merged

    # Plain scalar
    for v in page_values:
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# FieldExtractor
# ---------------------------------------------------------------------------

# Import jsonschema lazily below; it is available (4.26.0 is installed).
# Validation is opt-in: only performed if the merged data has at least one field
# value so we don't report spurious errors for completely empty pages.

_EXTRACTION_STATUS_MAP = {
    # docling ConversionStatus value  ->  our status string
    "success": "extracted",
    "partial_success": "partial",
    "failure": "extraction_failed",
    "skipped": "extraction_skipped",
    # catch-all for unexpected values
}


def _docling_status_to_ours(docling_status: str) -> str:
    return _EXTRACTION_STATUS_MAP.get(docling_status, "extraction_failed")


class FieldExtractor:
    """Run Docling ``DocumentExtractor`` on a file and return a merged result dict.

    The ``DocumentExtractor`` (and therefore torch/transformers) is loaded **lazily** on the first
    call to :meth:`run`.  Constructing a ``FieldExtractor`` is cheap.

    :param schema: JSON Schema as a dict (already loaded/validated).
    :param validate_output: If ``True`` (default), validate the merged record against the JSON
        Schema using ``jsonschema`` and append any violations to ``errors``.  Data is never
        discarded on validation failure.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        validate_output: bool = True,
    ) -> None:
        self._schema = schema
        self._template = schema_to_template(schema)
        self._schema_hash = _schema_hash(schema)
        self._validate_output = validate_output
        self._extractor: Any = None  # lazily initialised

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            # Deferred import so the module loads without pulling in torch
            from docling.datamodel.base_models import InputFormat
            from docling.document_extractor import DocumentExtractor

            self._extractor = DocumentExtractor(
                allowed_formats=[InputFormat.PDF, InputFormat.IMAGE]
            )
        return self._extractor

    def run(self, path: str | Path) -> dict[str, Any]:
        """Extract fields from *path* and return a result dict.

        Never raises — all errors are captured in ``result["errors"]`` and
        ``result["status"]`` is set to ``"extraction_failed"``.
        """
        p = Path(path)
        ext = p.suffix.lower()

        # Fast path: unsupported format — never loads the model
        if ext not in _EXTRACTABLE_EXTENSIONS:
            logger.info(
                "Field extraction skipped for %s (unsupported extension %r).", p.name, ext
            )
            return self._result("extraction_skipped", {}, [], [])

        try:
            return self._run_extraction(p)
        except Exception as exc:
            logger.exception("Field extraction failed for %s.", p.name)
            return self._result("extraction_failed", {}, [], [str(exc)])

    def _run_extraction(self, p: Path) -> dict[str, Any]:
        extractor = self._get_extractor()

        result = extractor.extract(p, template=self._template, raises_on_error=False)

        # Map Docling ConversionStatus to our status string
        our_status = _docling_status_to_ours(result.status.value)

        # Collect document-level errors
        doc_errors: list[str] = [
            err.error_message for err in result.errors if hasattr(err, "error_message")
        ]

        # Collect per-page data and errors
        raw_pages: list[dict[str, Any]] = []
        all_page_data: list[dict[str, Any] | None] = []

        for page in result.pages:
            raw_pages.append(
                {
                    "page_no": page.page_no,
                    "extracted_data": page.extracted_data,
                    "raw_text": page.raw_text,
                    "errors": list(page.errors),
                }
            )
            all_page_data.append(page.extracted_data)
            if page.errors:
                doc_errors.extend(page.errors)

        # If no pages had any extracted_data, treat as failed unless already skipped
        if all_page_data and all(d is None for d in all_page_data):
            if our_status not in ("extraction_skipped", "extraction_failed"):
                our_status = "extraction_failed"
            doc_errors.append("All pages returned null extracted_data (possible model or format issue).")

        # Merge pages
        merged = merge_pages(all_page_data, self._template, self._schema)

        # Optional jsonschema validation (non-fatal)
        if self._validate_output and any(v is not None for v in merged.values()):
            validation_errors = self._validate(merged)
            doc_errors.extend(validation_errors)

        return self._result(our_status, merged, raw_pages, doc_errors)

    def _validate(self, data: dict[str, Any]) -> list[str]:
        """Validate *data* against the schema; return a list of error strings."""
        try:
            import jsonschema

            validator = jsonschema.Draft7Validator(self._schema)
            return [
                f"Validation: {err.message} (path: {'/'.join(str(p) for p in err.absolute_path)})"
                for err in validator.iter_errors(data)
            ]
        except Exception as exc:  # noqa: BLE001 — guard dynamic import + schema build failures
            return [f"Validation error: {exc}"]

    def _result(
        self,
        status: str,
        data: dict[str, Any],
        pages: list[dict[str, Any]],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "status": status,
            "data": data,
            "pages": pages,
            "errors": errors,
            "schema_hash": self._schema_hash,
        }
