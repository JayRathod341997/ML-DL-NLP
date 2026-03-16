from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float


class NERModel:
    """HuggingFace NER pipeline with optional spaCy fallback for extended entity types.

    Primary model: dslim/bert-base-NER (PER, ORG, LOC, MISC)
    spaCy model: en_core_web_sm (DATE, MONEY, CARDINAL, GPE, PRODUCT, etc.)
    """

    HF_LABELS = {"PER", "ORG", "LOC", "MISC"}

    def __init__(
        self,
        hf_model: str = "dslim/bert-base-NER",
        use_spacy: bool = True,
        device: int = -1,  # -1 = CPU
    ) -> None:
        from transformers import pipeline

        self._hf_pipe = pipeline(
            "ner",
            model=hf_model,
            aggregation_strategy="simple",
            device=device,
        )
        self._spacy_nlp = None
        if use_spacy:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy en_core_web_sm not found. Run: uv run python -m spacy download en_core_web_sm")

    def predict(self, text: str) -> list[Entity]:
        """Extract entities from a text string.

        Returns a merged list of entities from HuggingFace NER and spaCy
        (spaCy used for entity types not covered by the HF model).
        """
        entities = self._predict_hf(text)
        if self._spacy_nlp:
            spacy_entities = self._predict_spacy(text)
            # Add spaCy entities for labels not in HF label set
            hf_spans = {(e.start, e.end) for e in entities}
            for ent in spacy_entities:
                if ent.label not in self.HF_LABELS and (ent.start, ent.end) not in hf_spans:
                    entities.append(ent)
        return sorted(entities, key=lambda e: e.start)

    def _predict_hf(self, text: str) -> list[Entity]:
        raw = self._hf_pipe(text)
        return [
            Entity(
                text=r["word"],
                label=r["entity_group"],
                start=r["start"],
                end=r["end"],
                score=float(r["score"]),
            )
            for r in raw
        ]

    def _predict_spacy(self, text: str) -> list[Entity]:
        doc = self._spacy_nlp(text)
        return [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                score=1.0,  # spaCy doesn't provide confidence scores
            )
            for ent in doc.ents
        ]
