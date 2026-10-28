"""LLM-based entity and relation extraction, used to build the document graph."""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an information-extraction engine building a knowledge graph \
from document chunks. Extract entities and the relations between them.

Return STRICT JSON (no markdown, no commentary) with this exact shape:
{{
  "entities":  [{{"name": "string", "type": "Person|Organization|Product|Location|Date|Concept|Metric|Document|Event|Other"}}],
  "relations": [{{"source": {{"name": "string", "type": "string"}}, "target": {{"name": "string", "type": "string"}}, "type": "RELATION_TYPE", "evidence": "short supporting quote"}}]
}}

Rules:
- Extract at most {max_entities} entities.
- Entity names must be canonical (e.g. "Acme Corp" not "Acme Corporation's").
- Only include relations explicitly supported by the text.
- Relation types: USES, MENTIONS, PART_OF, WORKS_FOR, LOCATED_IN, OWNS, PRODUCES, DEPENDS_ON, \
SIMILAR_TO, CONTRADICTS, REFERENCES, MEASURED_BY, OCCURRED_AT, OTHER.

Document title: {title}
Chunk text:
---
{chunk}
---"""


class EntityExtractor:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — entity extraction requires it.")
        self.client = OpenAI(api_key=s.openai_api_key, timeout=s.openai_timeout)
        self.model = s.openai_model
        self.max_entities = s.max_entities_per_chunk

    def extract(self, title: str, chunk_text: str) -> dict[str, list[dict[str, Any]]]:
        """Extract {"entities": [...], "relations": [...]} from a chunk."""
        prompt = EXTRACTION_PROMPT.format(
            max_entities=self.max_entities, title=title, chunk=chunk_text[:6000]
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(self._strip_json_fences(content))
        except Exception:
            logger.exception("Entity extraction failed for a chunk; skipping.")
            return {"entities": [], "relations": []}

        entities = [
            {"name": str(e.get("name", "")).strip(), "type": str(e.get("type", "Other"))}
            for e in data.get("entities", [])
            if e.get("name")
        ]
        valid_names = {(e["name"], e["type"]) for e in entities}
        relations = [
            r
            for r in data.get("relations", [])
            if (r.get("source", {}).get("name"), r.get("source", {}).get("type")) in valid_names
            and (r.get("target", {}).get("name"), r.get("target", {}).get("type")) in valid_names
        ]
        return {"entities": entities, "relations": relations}

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
