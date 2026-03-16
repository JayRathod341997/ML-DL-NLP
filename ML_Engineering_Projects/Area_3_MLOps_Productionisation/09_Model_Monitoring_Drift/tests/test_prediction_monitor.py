import pytest
from src.prediction_monitor import PredictionMonitor


@pytest.fixture
def monitor():
    return PredictionMonitor(window_size=5)


def test_update_returns_snapshot(monitor):
    snap = monitor.update(["POSITIVE", "NEGATIVE", "POSITIVE"])
    assert snap.num_predictions == 3
    assert "POSITIVE" in snap.label_distribution
    assert snap.entropy >= 0


def test_distribution_sums_to_one(monitor):
    snap = monitor.update(["POSITIVE"] * 3 + ["NEGATIVE"] * 7)
    total = sum(snap.label_distribution.values())
    assert abs(total - 1.0) < 1e-4


def test_entropy_uniform_is_max(monitor):
    snap_uniform = monitor.update(["A", "B", "C", "D"])
    snap_skewed = monitor.update(["A"] * 10)
    assert snap_uniform.entropy > snap_skewed.entropy


def test_history_df(monitor):
    for _ in range(3):
        monitor.update(["POSITIVE", "NEGATIVE"])
    df = monitor.get_history_df()
    assert len(df) == 3
    assert "entropy" in df.columns


def test_window_size_respected(monitor):
    for _ in range(10):
        monitor.update(["A", "B"])
    df = monitor.get_history_df()
    assert len(df) <= 5  # window_size=5
