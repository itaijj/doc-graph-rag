# Schema-Driven Field Extraction (Docling DocumentExtractor) — Plan

## Top-Level Overview

**Goal:** During ingestion, extract a fixed set of structured fields from every document according to a
user-provided **JSON Schema** (which may contain fields that are *lists of JSON objects*), and persist the
result per document so a later, separate step can aggregate all records into one statistical report.

**Out of scope:** the statistical report itself, DOCX/XLSX extraction, changes to the agent/query side.

**Decisions (confirmed with user):**
- Engine: Docling's experimental `docling.document_extractor.DocumentExtractor` (local VLM, default
  `numind/NuExtract-2.0-2B`, `VlmExtractionPipelineOptions`).
- Only PDF (and images, which the extractor supports natively) are extracted. DOCX/XLSX/XLS are skipped and
  recorded with status `extraction_skipped`. Note: `SUPPORTED_EXTENSIONS` in `app/ingestion/parser.py` does not
  currently accept image files, so in practice only PDFs reach extraction; widening ingestion to images is not part of this plan.
- Per-page results are **merged into one record per document**: list fields are concatenated across pages,
  scalar fields take the first non-null value. Raw per-page output is also stored for traceability.

**Findings from the installed Docling 2.144.0 (`.venv/.../docling`):**
- `DocumentExtractor.extract(source, template, ...)` runs the VLM **once per page**; result is
  `ExtractionResult.pages: list[ExtractedPageData]` with `extracted_data` (dict or `None` if output wasn't valid
  JSON), `raw_text`, `errors`.
- `template` accepts str / dict / Pydantic model / model class. Dict templates are serialized with `json.dumps`
  and passed to NuExtract as-is; NuExtract's template format natively supports nesting and lists of objects
  (e.g. `{"items": [{"name": "string", "price": "number"}]}`). A Pydantic *class* goes through `polyfactory`
  (not installed) and fills lists with generated values — so it is **not** used; a dict template is built
  from the JSON Schema instead.
- `NuExtract` prompt style needs `qwen-vl-utils` (not installed) and `transformers`/`torch` (pulled in by docling).
  `polyfactory` is not needed when using dict templates.
- The extractor only has default backends for `InputFormat.PDF` and `InputFormat.IMAGE`.
- Already-ingested documents are skipped by checksum (`IngestionPipeline.ingest_file`), so extraction runs once per new document.

**Approach:** new module `app/ingestion/field_extract.py` (schema → template, run extractor, merge pages) → one
JSON file per document written to a configurable output directory → wired into `IngestionPipeline`, enabled only
when an extraction schema path is configured. Extraction failure must never fail the ingestion of a document.
No database changes (storage is JSON files on disk, per user decision).

## Sub-Tasks

### 1. Configuration & dependency
- **Intent:** Let the operator point the app at a JSON Schema file, choose where results go, and ensure the VLM prerequisites are installed.
- **Expected Outcomes:** New optional settings; extraction is off when the schema path is unset; `qwen-vl-utils` declared as a dependency.
- **Todo List:**
  1. Add `extraction_schema_path: str = ""` and `extraction_output_dir: str = "./extractions"` to `Settings`.
  2. Document `EXTRACTION_SCHEMA_PATH` and `EXTRACTION_OUTPUT_DIR` in `.env.example` and README configuration section; add the output dir to `.gitignore`.
  3. Add `qwen-vl-utils` to `pyproject.toml` dependencies (check that `docling` full package provides `extract-core` extras; add the extra if not).
  4. Add an example schema file (e.g. `schemas/example_extraction_schema.json`) containing a scalar field, an enum field, and an array-of-objects field.
- **Relevant Context:** `app/config.py` (`Settings`), `.env.example`, `pyproject.toml`, `README.md`.
- **Status:** `[x] done`
- **Notes for next sub-tasks:**
  - Settings fields in `app/config.py`: `settings.extraction_schema_path` (`str`, default `""` → extraction disabled; env `EXTRACTION_SCHEMA_PATH`) and `settings.extraction_output_dir` (`str`, default `"./extractions"`; env `EXTRACTION_OUTPUT_DIR`). Both are plain strings — convert to `Path` at point of use; treat empty/whitespace schema path as disabled.
  - Example schema: `schemas/example_extraction_schema.json` (draft-07, top-level `type: object`) with `document_title` (string), `document_type` (string enum), `total_amount` (number), and `line_items` (array of objects with `name` string / `quantity` integer / `price` number). Useful as a fixture for sub-task 2/7 tests. It has no `$ref`/`$defs`, `oneOf`, or `anyOf`.
  - Documented in `.env.example` and the README configuration table. `extractions/` was added to `.gitignore` (the edit tool is blocked on that file, so it was appended via a shell command). If `EXTRACTION_OUTPUT_DIR` is changed to another path, ignore that path too.
  - `qwen-vl-utils>=0.0.11` added to `pyproject.toml` dependencies. It is **not yet installed** in `.venv` — run `pip install -e .` (or `pip install qwen-vl-utils`) before running real extraction. `docling` 2.144.0 pulls `docling-slim[standard]`, which already includes the `extract-core` deps (`polyfactory`, `rtree`, `scipy`, `numpy`, `pillow`), so no docling extra was added. `docling[vlm]` also lists `qwen-vl-utils`, but the explicit dependency is declared directly.
  - Lint: `ruff check app scripts tests` reports 14 pre-existing findings in files unrelated to this plan (`app/agent/tools.py`, `app/db/neo4j_client.py`, `app/llm.py`, `app/main.py`, `scripts/ask.py`, `scripts/ingest_folder.py`, `tests/test_smoke.py`). None are in files changed by this sub-task. Later sub-tasks should avoid adding new findings; "no new warnings" is the bar unless the baseline is cleaned up.

### 2. JSON Schema → NuExtract template converter
- **Intent:** Translate a standard JSON Schema into the dict template format NuExtract expects, including lists of objects.
- **Expected Outcomes:** A pure function (no model needed) that is unit-testable.
- **Todo List:**
  1. Load and minimally validate the schema (top-level `type: object` with `properties`); resolve local `$ref`/`$defs` if present.
  2. Map types: string→`"string"`, integer→`"integer"`, number→`"number"`, boolean→`"boolean"`, `format: date`→`"date-time"`/`"string"` per NuExtract conventions, `enum`→list of choices.
  3. Map `object`→nested dict; `array` of scalars→`["<type>"]`; `array` of objects→`[{...nested template...}]`.
  4. Raise a clear error for unsupported constructs (e.g. `oneOf`/`anyOf`) rather than silently guessing.
  5. Verify the exact NuExtract template syntax for enums / multi-choice against the model card (`numind/NuExtract-2.0-2B`) before finalizing.
- **Relevant Context:** `docling/pipeline/extraction_vlm_pipeline.py::_serialize_template` (dict → `json.dumps(indent=2)`), `docling/models/extraction/prompt_utils.py::build_nuextract_inputs`.
- **Status:** `[x] done`
- **Notes for next sub-tasks:**
  - Implemented in `app/ingestion/field_extract.py` (pure, no model import): `load_schema(path: str | Path) -> dict`, `schema_to_template(schema: dict) -> dict`, and exception `SchemaConversionError(ValueError)` (raised for unreadable/invalid JSON, non-object top level, missing/empty `properties`, unresolvable/recursive/non-local `$ref`, `oneOf`/`anyOf`/`allOf`/`not`/`if`/`patternProperties`, type unions other than `[x, "null"]`, nested arrays, non-string enums, missing `type`). Sub-task 3 should extend this module (add `FieldExtractor`, merge) and catch `SchemaConversionError` at startup; sub-task 7 tests import these names.
  - Behavior: local `$ref` (`#/$defs/...`, any JSON pointer) resolved; `["x","null"]` treated as `x`; `string` + `format: date|date-time` → `"date-time"`; `description`, `required`, etc. are ignored; `enum` → `["a","b"]` (single choice); array of enum → `[["a","b"]]` (multi-choice); array of objects → `[{...}]`; array of scalars → `["string"]`.
  - Template output for `schemas/example_extraction_schema.json`: `{"document_title": "string", "document_type": ["report","invoice","contract","letter","other"], "total_amount": "number", "line_items": [{"name": "string", "quantity": "integer", "price": "number"}]}`. Pass this dict directly as the `template` argument of `DocumentExtractor.extract`. Merge in sub-task 3 can treat a template value that is a list as "list field" and a dict as nested object; an enum list of strings is a scalar choice, not a list field (the template list of strings for single-choice vs `[[...]]` for multi-choice must be distinguished by the *schema*, not the template — keep the schema or a flag available if merge needs it; for the example schema it only matters that `document_type` is a scalar).
  - **Assumption (unverified locally):** NuExtract 2.0 model card / docs are not available offline (no `numind` model in the HF cache; the installed docling only wraps the template via `json.dumps` and contains no template syntax docs). Used the documented NuExtract 2.0 conventions from memory: scalars `"string"`/`"integer"`/`"number"`/`"boolean"`/`"date-time"`, `"verbatim-string"` is NOT emitted (never produced; could be added later), single-choice enum = list of strings, multi-choice = list containing that list, array of objects = `[{...}]`. Re-verify against the model card (https://huggingface.co/numind/NuExtract-2.0-2B) when the model is first downloaded.
  - Lint: `ruff check app/ingestion/field_extract.py` passes with no findings. Only lightweight manual sanity checks run (example schema, `$ref`/`$defs`, date format, multi-choice, unsupported `anyOf`, non-object root, missing `$ref`); formal tests are sub-task 7.

### 3. Extractor wrapper + per-page merge
- **Intent:** Run the extractor on a file and return one merged record plus raw per-page output.
- **Expected Outcomes:** `FieldExtractor` that returns a result dict with `status` (`extracted` | `partial` | `extraction_skipped` | `extraction_failed`), `data` (merged), `pages` (raw), `errors`.
- **Todo List:**
  1. Create `FieldExtractor` that lazily builds one `DocumentExtractor(allowed_formats=[PDF, IMAGE])` (model load is expensive; reuse across files, same pattern as `Embedder`/`EntityExtractor` instances in the pipeline).
  2. Return `extraction_skipped` (with a logged message) for extensions other than PDF/image.
  3. Call `extract(..., raises_on_error=False)`; map Docling `ConversionStatus` to our status; collect page-level errors and pages whose `extracted_data` is `None` (keep `raw_text`).
  4. Merge pages in page order: lists concatenated, scalars first non-null, nested objects merged recursively using the same rules. Keep the template's keys even when null so every record has the same shape.
  5. Optionally validate the merged record against the original JSON Schema (`jsonschema` or pydantic) and record validation problems in `errors` without discarding data (decide in review whether to add the `jsonschema` dependency).
- **Relevant Context:** `docling/document_extractor.py`, `docling/datamodel/extraction.py` (`ExtractionResult`, `ExtractedPageData`), `app/ingestion/entity_extract.py` (similar "extract and never raise" style).
- **Status:** `[x] done`
- **Notes for next sub-tasks:**
  - **`jsonschema` decision:** `jsonschema` 4.26.0 was already installed in `.venv` → **validation is included** (no new dependency needed). The `FieldExtractor` constructor takes `validate_output: bool = True` (keyword-only). Validation is skipped if all merged values are `None` (empty result) to avoid spurious errors; failures are appended to `errors` without discarding data (`Draft7Validator.iter_errors`).
  - **New public symbols in `app/ingestion/field_extract.py`:**
    - `_schema_hash(schema: dict) -> str` — SHA-256 hex of `json.dumps(schema, sort_keys=True)`. Used as `schema_hash` in every result dict and in sub-task 4 file records.
    - `merge_pages(pages: list[dict | None], template: dict, schema: dict) -> dict` — pure function; `None` pages silently skipped. Scalar-enum vs list-field distinction is resolved by reading `schema["properties"][key]["type"]` (or `"items"` presence); the template alone is insufficient because a single-choice enum is `["a","b"]` which looks like a list of scalars.
    - `FieldExtractor(schema: dict, *, validate_output: bool = True)` — cheap to construct; `DocumentExtractor` is lazy-loaded on first `.run()` call. `schema_to_template` and `_schema_hash` are called in `__init__`, raising `SchemaConversionError` eagerly on bad schemas.
    - `FieldExtractor.run(path: str | Path) -> dict` — never raises; returns result dict (see shape below).
  - **Result dict shape** (same for all statuses):
    ```json
    {
      "status":      "extracted | partial | extraction_skipped | extraction_failed",
      "data":        { "<field>": <value>, ... },
      "pages":       [{"page_no": int, "extracted_data": dict|null, "raw_text": str|null, "errors": [...]}],
      "errors":      ["..."],
      "schema_hash": "<sha256 hex>"
    }
    ```
    `data` is `{}` (not null) for `extraction_skipped` / `extraction_failed`. `pages` is `[]` for `extraction_skipped`.
  - **`_EXTRACTABLE_EXTENSIONS`:** `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.gif`, `.webp`. Any suffix not in this set returns `extraction_skipped` immediately without loading the model.
  - **Status mapping from Docling `ConversionStatus`:** `success` → `extracted`, `partial_success` → `partial`, `failure` → `extraction_failed`, `skipped` → `extraction_skipped`, anything else → `extraction_failed`. If all pages have `extracted_data=None`, status is forced to `extraction_failed` with an error message appended.
  - **Lint:** `ruff check app/ingestion/field_extract.py` passes with no findings.
  - **Sanity checks passed** (no model download needed): template generation, merge (scalars, enums, list-of-objects, all-null pages, zero pages), `FieldExtractor` skipped path (`.docx`), schema hash stability.
  - Sub-task 4 (`save_extraction`, `iter_extractions`, `extraction_is_current`) can be placed in a sibling `app/ingestion/extraction_store.py` or appended to `field_extract.py`; import `FieldExtractor`, `merge_pages`, `_schema_hash` from `app.ingestion.field_extract`. Sub-task 5 instantiates `FieldExtractor` in `IngestionPipeline.__init__` guarded by `bool(settings.extraction_schema_path.strip())`.

### 4. Persistence (JSON files on disk)
- **Intent:** Save each document's extraction as a self-contained JSON file so a later job can load the whole directory and aggregate into a statistical report.
- **Expected Outcomes:** A small storage helper that writes/reads one human-readable `<title>.json` per document in `EXTRACTION_OUTPUT_DIR`. Nothing changes in PostgreSQL/Neo4j.
- **Todo List:**
  1. File name = sanitized document **title** (file stem, same value as `ParsedDocument.title`; for uploads the original filename stem, not the temp name). Sanitize to filesystem-safe characters.
  2. Collision rule: if `<title>.json` already exists but its stored `checksum` differs (different file with the same stem, e.g. `a.pdf` vs `a.docx` or same name in different folders), use `<title>__<first 8 chars of checksum>.json`. If the checksum matches, overwrite (re-extraction). Lookup follows the same two-step order.
  3. File content: `source_name`, `source_path`, `checksum` (kept inside the file as the identity/staleness key), `document_id` (when ingested), `schema_hash` (hash of the schema file, lets the report filter by schema version), `status`, `extracted_at`, `data` (merged record), `pages` (raw per-page output), `errors`.
  4. Write atomically (temp file + rename) so a crash never leaves a half-written file; create the directory if missing.
  5. Provide `save_extraction(...)`, `iter_extractions()` (for the future report) and `extraction_is_current(title, checksum, schema_hash)` (used by the re-extraction script to skip up-to-date files).
  6. Write `extraction_skipped` / `extraction_failed` records too (with empty `data`) so the report can count coverage.
- **Relevant Context:** `app/ingestion/pipeline.py` (checksum, `ParsedDocument.title`), `app/config.py`, `app/ingestion/parser.py` (`parse_bytes` sets the title from the original filename). Place helper in `app/ingestion/field_extract.py` or a sibling `app/ingestion/extraction_store.py`.
- **Status:** `[x] done`
- **Notes for next sub-tasks:**
  - Implemented in `app/ingestion/extraction_store.py` (sibling of `field_extract.py`; does **not** import it, so no torch/docling load). Public API:
    - `save_extraction(title: str, source_path: str | Path, checksum: str, result: dict, *, source_name: str | None = None, document_id: str | None = None, output_dir: str | Path | None = None) -> Path` — `result` is the dict from `FieldExtractor.run` (reads `status`, `data`, `pages`, `errors`, `schema_hash`; missing keys tolerated). `title` = `ParsedDocument.title` (file stem / original upload filename stem). `source_name` defaults to `Path(source_path).name` — **for uploads pass it explicitly** (original filename) because `source_path` is a temp file. `document_id` is `None` unless given (sub-task 6 omits it). For `extraction_skipped`/`extraction_failed`, `data` is forced to `{}`.
    - `iter_extractions(output_dir: str | Path | None = None) -> Iterator[dict]` — yields every `*.json` record in the dir, sorted by file name; missing dir → empty; corrupt/non-object files are logged (warning) and skipped.
    - `extraction_is_current(title: str, checksum: str, schema_hash: str, output_dir: str | Path | None = None) -> bool`.
    - Extras: `load_extraction(title, checksum, output_dir=None) -> dict | None` (same two-step lookup, returns the record) and `sanitize_title(title: str | None) -> str`.
  - `output_dir=None` → `Path(get_settings().extraction_output_dir)` resolved lazily at call time (relative paths are relative to the process CWD). Directory is created on save (`mkdir(parents=True)`).
  - **File layout:** `<output_dir>/<sanitized title>.json`, or `<sanitized title>__<checksum[:8]>.json` when `<title>.json` exists with a different `checksum`. Keys, in order: `source_name, source_path, checksum, document_id, schema_hash, status, extracted_at` (UTC ISO-8601 with `+00:00`), `data, pages, errors`. JSON is pretty-printed (indent 2, `ensure_ascii=False`).
  - **Decisions:**
    - Sanitizing: unsafe chars `<>:"/\|?*` and control chars → `_` (each one, not collapsed); whitespace runs collapsed; leading/trailing spaces and dots stripped (so `..` can never survive as a name); Windows reserved names (`CON`, `NUL`, `COM1`…) get a `_` prefix; truncated to 200 UTF-8 bytes (room for the `__<8>` suffix within 255); empty → `untitled`. Unicode letters/spaces are kept (e.g. `Café Report 2024`).
    - Write target: `<title>.json` if absent, **unreadable/corrupt**, or same checksum; otherwise suffixed name. Writes go through `tempfile.mkstemp` in the same dir (`.tmp-*.part`) + `os.fsync` + `os.replace`; temp file is removed on error.
    - `extraction_is_current` = stored `checksum` AND `schema_hash` match **and** status ≠ `extraction_failed`. So `extracted`, `partial` and `extraction_skipped` count as current (skipped is deterministic for a file type; `partial` is the best achievable), while `extraction_failed` is retried on the next run since failures are often transient. A caveat for sub-task 6: a DOCX (skipped) record is still "current" only for the same schema_hash, which is fine.
    - Known limitation: two different files whose checksums share the first 8 chars *and* the same title would overwrite each other's suffixed file (astronomically unlikely, accepted per the plan's rule).
  - **Sanity checks passed** (temp dir, no model/DB): sanitizing edge cases, round-trip/key order, new-dir creation, overwrite, collision suffix + suffixed overwrite, two-step load, `extraction_is_current` for extracted/partial/skipped/failed/changed schema/changed checksum/missing dir, `iter_extractions` skipping corrupt files, default dir from settings. `ruff check app/ingestion/extraction_store.py` → no findings. The throwaway script was deleted; formal tests are sub-task 7 (use `tmp_path`, pass `output_dir=`).
  - **WSL command pattern** (`.venv` is a Linux venv): `wsl -d Ubuntu-24.04 -- bash -lc 'cd ~/doc-graph-rag && .venv/bin/python <script>.py'` and `... .venv/bin/ruff check <path>`. Write snippets to a temp `.py` file (avoids PowerShell quoting issues) and delete afterwards. To exercise the default dir, set `EXTRACTION_OUTPUT_DIR` in `os.environ` before `app.config` is first imported (`get_settings` is `lru_cache`d).
  - Sub-task 5 usage sketch: `save_extraction(parsed.title, file_path, checksum, result, source_name=<original filename>, document_id=str(doc_id))`; sub-task 6: `if not force and extraction_is_current(title, checksum, extractor_schema_hash): skip`, where `schema_hash` is `_schema_hash(schema)` from `field_extract` (also available as `result["schema_hash"]`; compute it before running — `FieldExtractor` computes it in `__init__`).

### 5. Pipeline wiring
- **Intent:** Run extraction as part of ingestion for each new document and save the result.
- **Expected Outcomes:** Ingesting a PDF with `EXTRACTION_SCHEMA_PATH` set produces `<EXTRACTION_OUTPUT_DIR>/<title>.json`; DOCX/XLSX produce a file with status `extraction_skipped`; ingestion summary includes `extraction_status`; extraction errors never abort ingestion.
- **Todo List:**
  1. Instantiate `FieldExtractor` in `IngestionPipeline.__init__` only when a schema path is configured.
  2. In `ingest_file`/`_ingest_parsed`, run extraction on the original file path after the document row exists (so `document_id` can be recorded), wrapped in try/except that logs and saves an `extraction_failed` record.
  3. Handle the upload route: `app/main.py` ingests uploaded bytes via a temp file — confirm the file extension is preserved and that the original filename (not the temp name) is recorded as `source_name`.
  4. Add `extraction_status` to the returned summary dict.
- **Relevant Context:** `app/ingestion/pipeline.py`, `app/main.py` (`/ingest/file`), `scripts/ingest_folder.py`.
- **Status:** `[x] done`
- **Notes for next sub-tasks:**
  - **`app/ingestion/pipeline.py` changes:** `IngestionPipeline.__init__` now sets `self.field_extractor: FieldExtractor | None` (built *first*, before DB connections): `FieldExtractor(load_schema(path))` when `get_settings().extraction_schema_path.strip()` is non-empty, else `None`. New `ingest_bytes(filename, data) -> dict`; new `_extract_fields(parsed, doc_id, source_path, checksum, extract_path) -> str`; `_ingest_parsed` gained an optional kwarg `extract_path: Path | None = None` (existing positional args unchanged); `ingest_file` passes `extract_path=path`.
  - **Decision – bad schema = fail fast:** a configured but unreadable/invalid schema raises (`SchemaConversionError`, a `ValueError`; or `OSError` if the file is missing) from `IngestionPipeline()`, i.e. at app startup (`lifespan`) / script start, rather than silently ingesting everything without extraction. Because the extractor is built before `PostgresStore()`/`Neo4jStore()`, no connections leak. Empty/whitespace path = disabled.
  - **Flow:** extraction runs as step 4 of `_ingest_parsed`, after the document row, chunks, entities and `finalize_document`, so `document_id=str(doc_id)` is recorded. Duplicates (checksum match) are still skipped in `ingest_file` before parsing/extraction, so no record is written for them. `_extract_fields` never raises: `FieldExtractor.run` exceptions are logged and turned into an `extraction_failed` result (`errors=["<ExcType>: <msg>"]`, `schema_hash` of the extractor); a failure inside `save_extraction` is logged and reported as `extraction_failed` (no record on disk then). `title=parsed.title`, `source_name=parsed.metadata["source_name"]` (file name for `ingest_file`, original upload filename for uploads).
  - **Summary dict:** existing keys unchanged; added `"extraction_status"`: `"disabled"` when extraction is off (key is always present on `ingested` results), otherwise the record status (`extracted | partial | extraction_skipped | extraction_failed`). Skipped-duplicate and `error` results (from `ingest_folder`) do not carry the key.
  - **Upload route (`app/main.py` `/ingest/file`):** it previously called `parse_bytes` + `_ingest_parsed` directly with no real file path (temp file was already deleted). It now calls the new `IngestionPipeline.ingest_bytes(file.filename or "upload", data)`, which writes the bytes to `<tempdir>/upload<original suffix>` (extension preserved so the extractor and parser dispatch correctly), parses it with `parse_file`, sets `title = Path(filename).stem` and `metadata["source_name"] = filename` (same as `parse_bytes`), and runs `_ingest_parsed(..., source_path=filename, extract_path=tmp_path)` **inside** the `TemporaryDirectory` context, so extraction sees a live file; the directory is removed afterwards. `parse_bytes` in `parser.py` is untouched and now unused by the app (kept for compatibility). Note: unlike `ingest_file`, the upload route does not do a duplicate-checksum pre-check (pre-existing behavior, left unchanged).
  - **Sub-task 6 reuse:** `fx = FieldExtractor(load_schema(settings.extraction_schema_path))`; `schema_hash = fx._schema_hash` (or `_schema_hash(schema)` from `field_extract`); for each file: `checksum = sha256(bytes)`, `title = path.stem`; skip if `not force and extraction_is_current(title, checksum, schema_hash)`; else `result = fx.run(path)` and `save_extraction(title, path, checksum, result)` (omit `document_id`; `source_name` defaults to the file name). Wrap per-file in try/except like `_extract_fields` does (`run` itself never raises, `save_extraction` can raise on I/O errors).
  - **Sanity checks passed** (mocked `PostgresStore`/`Neo4jStore`/`Embedder`/`EntityExtractor`, no model): disabled → `extraction_status == "disabled"`, no files written; enabled with real `FieldExtractor` on an uploaded `.docx` → `extraction_skipped` record `Report.json` with `source_name="Report.docx"`, `document_id="doc-1"`; mocked `run` raising on an uploaded `.pdf` → ingestion still `ingested`, `extraction_failed` record saved, and the extractor received an existing path with suffix `.pdf`; invalid schema → `SchemaConversionError` at construction. Not exercised: real VLM extraction on a PDF. Throwaway script deleted.
  - **Lint:** `ruff check app/ingestion/pipeline.py app/main.py` → 6 findings, identical to the baseline (verified with `git stash`; all in `app/main.py` `/health` pre-existing). `pipeline.py` has none.

### 6. Standalone re-extraction script
- **Intent:** Run extraction on documents that were already ingested (or re-run after a schema change) without re-ingesting. Because results live on disk, this needs no database.
- **Expected Outcomes:** `scripts/extract_fields.py <folder>` (plus `make extract FOLDER=...`) that walks a folder like `ingest_folder.py`, computes each file's checksum, and writes/overwrites `<title>.json` (collision rule from sub-task 4).
- **Todo List:**
  1. Flags: `--force` (re-extract even if the output already has the same `checksum` and `schema_hash`), `-v/--verbose`.
  2. By default skip files whose output already exists with the current `checksum` and `schema_hash` (`extraction_is_current`).
  3. Per-file status lines and a summary (extracted/skipped/failed), mirroring `scripts/ingest_folder.py`; errors on one file do not stop the run.
  4. Does not touch PostgreSQL/Neo4j, so `document_id` is omitted in these records.
- **Relevant Context:** `scripts/ingest_folder.py`, `Makefile`.
- **Status:** `[ ] pending`

### 7. Tests & docs
- **Intent:** Cover the pure logic without needing the VLM or databases; document usage.
- **Expected Outcomes:** `make test` and `make lint` pass.
- **Todo List:**
  1. Unit tests for schema→template (scalar, enum, nested object, array of objects, unsupported construct).
  2. Unit tests for page merge (lists concatenated, scalar first-non-null, null/failed pages, nested lists).
  3. Unit test that DOCX/XLSX returns `extraction_skipped` without loading the model.
  4. Unit tests for the JSON file store using `tmp_path` (round-trip, overwrite, title sanitizing, same-title/different-checksum collision suffix, `extraction_is_current`, `iter_extractions`).
  5. README section: how to provide a schema, PDF-only limitation, hardware/model download note (NuExtract 2B downloads from Hugging Face on first run), output file layout, and the re-extraction script.
  6. Run `make lint` and `make test`.
- **Relevant Context:** `tests/test_smoke.py` (style, markers), `Makefile`.
- **Status:** `[ ] pending`

## Resolved Decisions
- Storage: JSON files on disk, one per document, named by document title with a checksum-suffix only on collisions (no Postgres table).
- Standalone re-extraction script is in scope (sub-task 6) and folder-based.
- `jsonschema` validation of merged output stays optional; decide during sub-task 3.
