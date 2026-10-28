#!/usr/bin/env python
"""CLI: ingest all supported documents in a folder (recursive).

Usage: python scripts/ingest_folder.py <folder>
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pipeline import IngestionPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of documents.")
    parser.add_argument("folder", type=Path, help="Folder to ingest recursively")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.folder.is_dir():
        parser.error(f"'{args.folder}' is not a directory")

    pipeline = IngestionPipeline()
    try:
        results = pipeline.ingest_folder(args.folder)
    finally:
        pipeline.close()

    for r in results:
        status = r.get("status")
        print(f"  [{status:>9}] {r.get('file', '?')}"
              + (f" — {r.get('num_chunks')} chunks, "
                 f"{r.get('num_entities')} entities, {r.get('num_relations')} relations"
                 if status == "ingested" else ""))
    print(f"\nDone: {len(results)} file(s), "
          f"{sum(1 for r in results if r.get('status') == 'ingested')} ingested, "
          f"{sum(1 for r in results if r.get('status') == 'skipped')} skipped, "
          f"{sum(1 for r in results if r.get('status') == 'error')} errors")


if __name__ == "__main__":
    main()
