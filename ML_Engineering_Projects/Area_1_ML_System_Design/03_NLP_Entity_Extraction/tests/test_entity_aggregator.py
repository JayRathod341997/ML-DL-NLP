import pytest
from src.ner_model import Entity
from src.entity_aggregator import EntityAggregator


@pytest.fixture
def aggregator():
    return EntityAggregator()


def test_aggregate_groups_same_text(aggregator):
    entities = [
        Entity("Apple", "ORG", 0, 5, 0.98),
        Entity("Apple", "ORG", 50, 55, 0.95),
        Entity("Apple", "ORG", 100, 105, 0.97),
    ]
    result = aggregator.aggregate(entities)
    assert len(result) == 1
    assert result[0].count == 3


def test_aggregate_majority_label(aggregator):
    entities = [
        Entity("Apple", "ORG", 0, 5, 0.9),
        Entity("Apple", "ORG", 20, 25, 0.9),
        Entity("Apple", "MISC", 40, 45, 0.5),
    ]
    result = aggregator.aggregate(entities)
    assert result[0].label == "ORG"


def test_aggregate_sorted_by_count_confidence(aggregator):
    entities = [
        Entity("Rare", "PER", 0, 4, 0.9),
        Entity("Common", "ORG", 10, 16, 0.9),
        Entity("Common", "ORG", 30, 36, 0.9),
        Entity("Common", "ORG", 50, 56, 0.9),
    ]
    result = aggregator.aggregate(entities)
    assert result[0].text == "Common"


def test_aggregate_empty_returns_empty(aggregator):
    assert aggregator.aggregate([]) == []
