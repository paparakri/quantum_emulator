import numpy as np
import pytest

from quantum_emulator.states import QuantumState
from quantum_emulator.gates import Gate, X, H

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
    np.array([1]),
    np.array([1, 0, 0, 0, 0, 0]),
    np.array([[1, 0], [0, 0]]),
    np.array([1, 1, 0, 0]),
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

class _NonSymmetricGate(Gate):
    name = "NonSymmetric"
    # Real orthogonal (3-4-5 rotation) and NOT symmetric: G.T != G
    matrix = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=np.complex128)

X = X()
H = H()
NonSymmetric = _NonSymmetricGate()

@pytest.mark.parametrize("target, expected", [
    (0, [0, 0, 1, 0]),
    (1, [0, 1, 0, 0]),
])
def test_x_ordering_two_qubits(target, expected):
    state = QuantumState("00")
    state.apply(X, target)
    np.testing.assert_allclose(state.qarray, expected)

def test_x_on_middle_qubit_of_three():
    state = QuantumState("000")
    state.apply(X, 1)
    expected = np.zeros(8)
    expected[0b010] = 1  # index 2
    np.testing.assert_allclose(state.qarray, expected)

# Superposition
def test_h_on_qubit_zero():
    state = QuantumState("00")
    state.apply(H, 0)
    s = 1 / np.sqrt(2)
    np.testing.assert_allclose(state.qarray, [s, 0, s, 0])

def test_h_on_both_qubits_uniform_probabilities():
    state = QuantumState("00")
    state.apply(H, 0)
    state.apply(H, 1)
    np.testing.assert_allclose(state.probabilities(), [1/4, 1/4, 1/4, 1/4])

# Non-symmetric gate: checks matrix-on-the-left and factor order
@pytest.mark.parametrize("target, expected", [
    (0, [0.6, 0, 0.8, 0]),   # G|0> (x) |0>
    (1, [0.6, 0.8, 0, 0]),   # |0> (x) G|0>
])
def test_non_symmetric_gate(target, expected):
    state = QuantumState("00")
    state.apply(NonSymmetric, target)
    np.testing.assert_allclose(state.qarray, expected)

def test_non_symmetric_gate_on_excited_state():
    # G|1> = [-0.8, 0.6]; qubit 1 of "01" -> |0> (x) G|1>
    state = QuantumState("01")
    state.apply(NonSymmetric, 1)
    np.testing.assert_allclose(state.qarray, [-0.8, 0.6, 0, 0])

# Norm preservation
def test_norm_preserved_after_several_gates():
    state = QuantumState("000")
    for gate, target in [(H, 0), (NonSymmetric, 1), (X, 2), (H, 1), (NonSymmetric, 0)]:
        state.apply(gate, target)
    assert np.isclose(np.linalg.norm(state.qarray), 1.0)

# Errors
@pytest.mark.parametrize("bad_gate", [np.eye(2), "X", None, 5])
def test_non_gate_raises_type_error(bad_gate):
    state = QuantumState("00")
    with pytest.raises(TypeError):
        state.apply(bad_gate, 0)

@pytest.mark.parametrize("bad_target", [-1, 2, 5, 1.5, "0", None])
def test_bad_target_raises_value_error(bad_target):
    state = QuantumState("00")  # valid targets: 0 and 1
    with pytest.raises(ValueError):
        state.apply(X, bad_target)