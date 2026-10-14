import numpy as np
import pytest

from quantum_emulator.states import Qubit, PLUS, MINUS


@pytest.mark.parametrize("init_val, expected", [
    (0, [1, 0]),
    (1, [0, 1]),
    (np.array([1,1])/np.sqrt(2), [1,1]/np.sqrt(2))
])
def test_basis_state_values(init_val, expected):
    np.testing.assert_allclose(Qubit(init_val).qarray, expected)


@pytest.mark.parametrize("init_val", [0, 1])
def test_basis_state_is_normalized(init_val):
    assert np.isclose(np.linalg.norm(Qubit(init_val).qarray), 1.0)


@pytest.mark.parametrize("init_val", [0, 1])
def test_basis_state_shape_and_dtype(init_val):
    state = Qubit(init_val).qarray
    assert state.shape == (2,)
    assert state.dtype == np.complex128


@pytest.mark.parametrize("bad_val", [2, -1, 5, "hello", None])
def test_invalid_init_val_raises(bad_val):
    with pytest.raises(ValueError):
        Qubit(bad_val)


@pytest.mark.parametrize("bad_array", [np.array([1, 0, 0]), np.array([[1], [0]])])
def test_wrong_shape_array_raises(bad_array):
    with pytest.raises(ValueError):
        Qubit(bad_array)


@pytest.mark.parametrize("bad_array", [np.array([0.5, 0.5]), np.array([1, 1])])
def test_unnormalized_array_raises(bad_array):
    with pytest.raises(ValueError):
        Qubit(bad_array)


def test_normalized_array_with_negative_amplitude_is_accepted():
    state = np.array([1, -1]) / np.sqrt(2)
    np.testing.assert_allclose(Qubit(state).qarray, state)


def test_float_array_is_converted_to_complex():
    q = Qubit(np.array([1.0, 0.0]))
    assert q.qarray.dtype == np.complex128


def test_qubit_does_not_share_memory_with_input_array():
    original = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    q = Qubit(original)
    original[0] = 0
    np.testing.assert_allclose(q.qarray, np.array([1, 1]) / np.sqrt(2))


@pytest.mark.parametrize("init_val, expected", [(0, 0), (1, 1)])
def test_measure_basis_state_is_deterministic(init_val, expected):
    assert Qubit(init_val).measure() == expected


@pytest.mark.parametrize("state", [PLUS, MINUS])
def test_measure_returns_valid_outcome(state):
    assert Qubit(state).measure() in (0, 1)


@pytest.mark.parametrize("state", [PLUS, MINUS])
def test_measure_collapses_to_observed_outcome(state):
    q = Qubit(state)
    outcome = q.measure()
    np.testing.assert_allclose(q.qarray, np.eye(2)[outcome])


@pytest.mark.parametrize("state", [PLUS, MINUS])
def test_measuring_twice_gives_same_result(state):
    q = Qubit(state)
    first = q.measure()
    assert q.measure() == first


def test_measure_is_reproducible_with_same_seed():
    first = Qubit(PLUS).measure(rng=np.random.default_rng(42))
    second = Qubit(PLUS).measure(rng=np.random.default_rng(42))
    assert first == second


def test_measure_plus_is_fifty_fifty():
    rng = np.random.default_rng(1234)
    shots = 1000
    ones = sum(Qubit(PLUS).measure(rng=rng) for _ in range(shots))
    sigma = np.sqrt(shots * 0.5 * 0.5)
    assert abs(ones - shots / 2) < 3 * sigma

SKEWED = np.array([np.sqrt(0.2), np.sqrt(0.8)])


@pytest.mark.parametrize("init_val, expected", [
    (0, {"0": 1000, "1": 0}),
    (1, {"0": 0, "1": 1000}),
])
def test_sample_basis_state_is_deterministic(init_val, expected):
    assert Qubit(init_val).sample(shots=1000) == expected


@pytest.mark.parametrize("state", [PLUS, MINUS, SKEWED])
def test_sample_counts_sum_to_shots(state):
    counts = Qubit(state).sample(shots=1234)
    assert counts["0"] + counts["1"] == 1234


@pytest.mark.parametrize("state, p0", [(PLUS, 0.5), (MINUS, 0.5), (SKEWED, 0.2)])
def test_sample_counts_within_three_sigma(state, p0):
    shots = 10_000
    counts = Qubit(state).sample(shots=shots, rng=np.random.default_rng(2024))
    sigma = np.sqrt(shots * p0 * (1 - p0))
    assert abs(counts["0"] - shots * p0) < 3 * sigma


def test_sample_is_reproducible_with_same_seed():
    first = Qubit(PLUS).sample(shots=500, rng=np.random.default_rng(7))
    second = Qubit(PLUS).sample(shots=500, rng=np.random.default_rng(7))
    assert first == second


def test_sample_does_not_change_state():
    q = Qubit(PLUS)
    q.sample(shots=1000)
    np.testing.assert_allclose(q.qarray, PLUS)


@pytest.mark.parametrize("bad_shots", [0, -5, 2.5])
def test_sample_rejects_invalid_shots(bad_shots):
    with pytest.raises(ValueError):
        Qubit(PLUS).sample(shots=bad_shots)