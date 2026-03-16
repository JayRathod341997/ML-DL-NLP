from __future__ import annotations

import csv
import json
from pathlib import Path

from .entity_aggregator import AggregatedEntity
from .relation_extractor import Relation


class OutputFormatter:
    """Serialises extracted entities and relations to JSON, CSV, or HTML."""

    def to_json(
        self,
        source: str,
        entities: list[AggregatedEntity],
        relations: list[Relation],
        output_path: str | Path | None = None,
    ) -> dict:
        data = {
            "source": source,
            "entity_count": len(entities),
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "count": e.count,
                    "confidence": e.confidence,
                    "variants": e.variants,
                }
                for e in entities
            ],
            "relations": [
                {
                    "entity1": r.entity1,
                    "entity2": r.entity2,
                    "relation": r.relation,
                    "count": r.count,
                }
                for r in relations
            ],
        }
        if output_path:
            Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    def to_csv(
        self,
        entities: list[AggregatedEntity],
        output_path: str | Path,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label", "count", "confidence"])
            writer.writeheader()
            for e in entities:
                writer.writerow(
                    {"text": e.text, "label": e.label, "count": e.count, "confidence": e.confidence}
                )

    def to_displacy_html(self, text: str, entities: list[AggregatedEntity]) -> str:
        """Generate a spaCy displacy-compatible HTML visualisation."""
        try:
            import spacy
            from spacy import displacy

            nlp = spacy.blank("en")
            doc = nlp.make_doc(text)
            spans = []
            for ent in entities:
                start_char = text.find(ent.text)
                if start_char == -1:
                    continue
                end_char = start_char + len(ent.text)
                span = doc.char_span(start_char, end_char, label=ent.label)
                if span:
                    spans.append(span)
            doc.ents = spans
            return displacy.render(doc, style="ent", page=True)
        except Exception as e:
            return f"<p>HTML rendering failed: {e}</p>"
