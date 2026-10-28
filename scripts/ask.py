#!/usr/bin/env python
"""CLI: ask the LangGraph agent a question.

Usage: python scripts/ask.py "your question here"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import DocGraphAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the doc-graph agent.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="print full result as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    agent = DocGraphAgent()
    try:
        result = agent.run(args.question)
    finally:
        agent.toolkit.close()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result["answer"])
        print(f"\n— {result['iterations']} planning round(s); "
              f"evidence from: {', '.join(result['evidence_sources']) or 'none'}")


if __name__ == "__main__":
    main()
