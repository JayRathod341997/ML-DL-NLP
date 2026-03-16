from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .ner_model import Entity


@dataclass
class AggregatedEntity:
    text: str           # canonical surface form (most frequent)
    label: str          # majority-vote label
    count: int          # total occurrences across all pages
    confidence: float   # mean confidence of all occurrences
    variants: list[str] # other surface forms seen (e.g., "Obama" + "Barack Obama")


class EntityAggregator:
    """Groups raw entities by normalised text, picks canonical label and form."""

    def aggregate(self, entities: list[Entity]) -> list[AggregatedEntity]:
        """Aggregate a flat list of Entity objects.

        Groups by lowercased text, resolves label via majority vote,
        ranks by (count × mean_confidence) descending.
        """
        groups: dict[str, list[Entity]] = defaultdict(list)
        for ent in entities:
            key = ent.text.strip().lower()
            groups[key].append(ent)

        aggregated = []
        for key, group in groups.items():
            canonical = max(
                set(e.text for e in group),
                key=lambda t: sum(1 for e in group if e.text == t),
            )
            label = self._majority_label(group)
            confidence = sum(e.score for e in group) / len(group)
            variants = list({e.text for e in group} - {canonical})
            aggregated.append(
                AggregatedEntity(
                    text=canonical,
                    label=label,
                    count=len(group),
                    confidence=round(confidence, 4),
                    variants=variants,
                )
            )

        return sorted(aggregated, key=lambda a: a.count * a.confidence, reverse=True)

    @staticmethod
    def _majority_label(group: list[Entity]) -> str:
        label_counts: dict[str, float] = defaultdict(float)
        for ent in group:
            label_counts[ent.label] += ent.score
        return max(label_counts, key=label_counts.__getitem__)
