from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .entity_aggregator import AggregatedEntity


@dataclass
class Relation:
    entity1: str
    entity2: str
    relation: str  # "co-occurrence" for simple; typed if RE model used
    count: int


class RelationExtractor:
    """Extracts relations between entities using co-occurrence within a window."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size  # character window for co-occurrence

    def extract_from_text(
        self, text: str, entities: list[AggregatedEntity]
    ) -> list[Relation]:
        """Find co-occurring entity pairs within a sliding character window.

        Args:
            text: Source document text.
            entities: Aggregated entities from the same document.
            window_size: Character window to consider two entities co-occurring.

        Returns:
            List of Relation objects sorted by count descending.
        """
        # Build a simple positional index from the raw text
        positions: dict[str, list[int]] = defaultdict(list)
        for ent in entities:
            idx = text.lower().find(ent.text.lower())
            while idx != -1:
                positions[ent.text].append(idx)
                idx = text.lower().find(ent.text.lower(), idx + 1)

        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        entity_names = list(positions.keys())
        for i, e1 in enumerate(entity_names):
            for e2 in entity_names[i + 1 :]:
                for p1 in positions[e1]:
                    for p2 in positions[e2]:
                        if abs(p1 - p2) <= self.window_size:
                            key = tuple(sorted([e1, e2]))
                            pair_counts[key] += 1

        relations = [
            Relation(entity1=k[0], entity2=k[1], relation="co-occurrence", count=v)
            for k, v in pair_counts.items()
        ]
        return sorted(relations, key=lambda r: r.count, reverse=True)
