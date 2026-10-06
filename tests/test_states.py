import numpy as np
import pytest

from quantum_emulator.states import Qubit


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
