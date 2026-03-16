import pytest
from src.ner_model import NERModel, Entity


@pytest.fixture(scope="module")
def ner():
    return NERModel(use_spacy=False)


def test_predict_returns_entities(ner):
    text = "Apple was founded by Steve Jobs in Cupertino, California."
    entities = ner.predict(text)
    assert len(entities) > 0
    assert all(isinstance(e, Entity) for e in entities)


def test_entity_labels_valid(ner):
    text = "Google and Microsoft are technology companies."
    entities = ner.predict(text)
    valid_labels = {"PER", "ORG", "LOC", "MISC"}
    for e in entities:
        assert e.label in valid_labels


def test_entity_scores_in_range(ner):
    text = "Barack Obama was the 44th President of the United States."
    entities = ner.predict(text)
    for e in entities:
        assert 0.0 <= e.score <= 1.0


def test_empty_text_returns_empty(ner):
    entities = ner.predict("")
    assert entities == []
