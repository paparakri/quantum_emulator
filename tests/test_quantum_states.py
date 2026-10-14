import numpy as np
import pytest

from quantum_emulator.states import QuantumState

@pytest.mark.parametrize("state, expected, num", [
    ("0", [1,0], 1),
    ("1", [0,1], 1),
    ("00", [1,0,0,0], 2),
    ("01", [0,1,0,0], 2),
    ("10", [0,0,1,0], 2),
    ("11", [0,0,0,1], 2),
    ("101", [0,0,0,0,0,1,0,0], 3)
    ])
def test_basic_quantum_states(state, expected, num):
    qubit = QuantumState(state)
    np.testing.assert_allclose(qubit.qarray, expected)
    assert qubit.num_qubits == num

@pytest.mark.parametrize("state", ["", "012", "abc", "601"])
def test_bad_quantum_states(state):
    with pytest.raises(ValueError):
        QuantumState(state)

def test_probabilities_quantum_state():
    qubits = QuantumState(np.array([1/2,1/2,1/2,1/2]))
    np.testing.assert_allclose(qubits.probabilities(), [1/4,1/4,1/4,1/4])

@pytest.mark.parametrize("length, num", [(2, 1), (4, 2), (8, 3)])
def test_array_sets_num_qubits(length, num):
    state = np.zeros(length)
    state[0] = 1
    assert QuantumState(state).num_qubits == num


@pytest.mark.parametrize("bad_array", [
    np.array([1]),                  # length 1: 2^0, not a valid register
    np.array([1, 0, 0, 0, 0, 0]),   # length 6: not a power of 2
    np.array([[1, 0], [0, 0]]),     # 2D
    np.array([1, 1, 0, 0]),         # unnormalized
])
def test_bad_arrays_raise(bad_array):
    with pytest.raises(ValueError):
        QuantumState(bad_array)


@pytest.mark.parametrize("bad_input", [5, None, 3.0])
def test_wrong_type_raises(bad_input):
    with pytest.raises(TypeError):
        QuantumState(bad_input)


@pytest.mark.parametrize("init", ["01", np.array([0.0, 1.0])])
def test_state_is_complex(init):
    assert QuantumState(init).qarray.dtype == np.complex128


def test_state_does_not_share_memory_with_input():
    original = np.array([1, 0, 0, 0], dtype=np.complex128)
    state = QuantumState(original)
    original[0] = 0
    np.testing.assert_allclose(state.qarray, [1, 0, 0, 0])