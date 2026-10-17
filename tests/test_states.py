import numpy as np
import pytest

from quantum_emulator.states import QuantumState, PLUS, MINUS

# Single-qubit (n=1) tests ported from the retired Qubit class.
# Anything already covered by test_quantum_states.py was dropped.

def test_plus_array_init():
    np.testing.assert_allclose(QuantumState(PLUS).qarray, np.array([1, 1]) / np.sqrt(2))


@pytest.mark.parametrize("bad_array", [np.array([0.5, 0.5]), np.array([1, 1])])
def test_unnormalized_array_raises(bad_array):
    with pytest.raises(ValueError):
        QuantumState(bad_array)


def test_normalized_array_with_negative_amplitude_is_accepted():
    state = np.array([1, -1]) / np.sqrt(2)
    np.testing.assert_allclose(QuantumState(state).qarray, state)


@pytest.mark.parametrize("state", [PLUS, MINUS])
def test_measure_collapses_to_observed_outcome(state):
    q = QuantumState(state)
    outcome = q.measure()
    assert outcome in ("0", "1")
    np.testing.assert_allclose(q.qarray, np.eye(2)[int(outcome, 2)])


def test_measure_is_reproducible_with_same_seed():
    first = QuantumState(PLUS).measure(rng=np.random.default_rng(42))
    second = QuantumState(PLUS).measure(rng=np.random.default_rng(42))
    assert first == second


def test_measure_plus_is_fifty_fifty():
    rng = np.random.default_rng(1234)
    shots = 1000
    ones = sum(QuantumState(PLUS).measure(rng=rng) == "1" for _ in range(shots))
    sigma = np.sqrt(shots * 0.5 * 0.5)
    assert abs(ones - shots / 2) < 3 * sigma

SKEWED = np.array([np.sqrt(0.2), np.sqrt(0.8)])


@pytest.mark.parametrize("state, p0", [(PLUS, 0.5), (MINUS, 0.5), (SKEWED, 0.2)])
def test_sample_counts_within_three_sigma(state, p0):
    shots = 10_000
    counts = QuantumState(state).sample(shots=shots, rng=np.random.default_rng(2024))
    sigma = np.sqrt(shots * p0 * (1 - p0))
    assert abs(counts["0"] - shots * p0) < 3 * sigma
