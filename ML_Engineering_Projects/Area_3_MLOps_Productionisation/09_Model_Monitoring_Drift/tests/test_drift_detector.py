import numpy as np
import pytest
from src.drift_detector import DriftDetector, DriftResult


@pytest.fixture
def detector():
    return DriftDetector(ks_threshold=0.05, psi_threshold=0.1)


def test_ks_same_distribution(detector):
    data = np.random.normal(0, 1, 1000)
    result = detector.ks_test(data, data + np.random.normal(0, 0.01, 1000))
    # Near-identical distributions should NOT trigger drift
    assert not result.is_drift


def test_ks_different_distribution(detector):
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(5, 1, 1000)  # very different mean
    result = detector.ks_test(ref, cur)
    assert result.is_drift
    assert result.p_value < 0.05


def test_psi_no_drift(detector):
    data = np.random.normal(0, 1, 2000)
    ref, cur = data[:1000], data[1000:]
    result = detector.psi(ref, cur)
    assert result.score < 0.1


def test_psi_high_drift(detector):
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(5, 2, 1000)  # large shift
    result = detector.psi(ref, cur)
    assert result.is_drift


def test_chi_squared_same_dist(detector):
    ref = {"POSITIVE": 500, "NEGATIVE": 500}
    cur = {"POSITIVE": 490, "NEGATIVE": 510}
    result = detector.chi_squared(ref, cur)
    assert not result.is_drift


def test_chi_squared_drift(detector):
    ref = {"POSITIVE": 500, "NEGATIVE": 500}
    cur = {"POSITIVE": 900, "NEGATIVE": 100}  # big shift
    result = detector.chi_squared(ref, cur)
    assert result.is_drift


def test_drift_result_fields(detector):
    result = detector.ks_test(np.zeros(100), np.ones(100))
    assert isinstance(result, DriftResult)
    assert result.test == "ks"
    assert result.score >= 0
    assert result.p_value is not None
