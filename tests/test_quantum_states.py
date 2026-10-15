import numpy as np
import pytest

from quantum_emulator.states import QuantumState
from quantum_emulator.gates import Gate, X as XGate, H as HGate, Z as ZGate

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

X = XGate()
H = HGate()
Z = ZGate()
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

@pytest.mark.parametrize("bits, expected1, expected2", [
        ("00", [1,0,0,0], [1,0,0,0]),
        ("01", [0,1,0,0], [0,0,0,1]),
        ("10", [0,0,0,1], [0,0,1,0]),
        ("11", [0,0,1,0], [0,1,0,0])
    ])
def test_controlled_gate(bits, expected1, expected2):
    state = QuantumState(bits)
    state2 = QuantumState(bits)
    state.cnot(0,1)
    state2.cnot(1,0)
    np.testing.assert_allclose(state.qarray,expected1)
    np.testing.assert_allclose(state2.qarray,expected2)

def test_bell_state():
    q = QuantumState("00")
    q.apply(H, 0)
    q.cnot(0,1)
    np.testing.assert_allclose(q.qarray, [1,0,0,1]/np.sqrt(2))

@pytest.mark.parametrize("bits, control, target, expected", [
        ("100", 0, 2, [0,0,0,0,0,1,0,0]),
        ("001", 2, 0, [0,0,0,0,0,1,0,0]),
        ("010", 0, 2, [0,0,1,0,0,0,0,0]),
    ])
def test_cnot_three_qubits(bits, control, target, expected):
    state = QuantumState(bits)
    state.cnot(control, target)
    np.testing.assert_allclose(state.qarray, expected)


@pytest.mark.parametrize("bits, expected", [
        ("00", [1,0,0,0]),
        ("01", [0,1,0,0]),
        ("10", [0,0,1,0]),
        ("11", [0,0,0,-1]),
    ])
def test_controlled_Z(bits, expected):
    state = QuantumState(bits)
    state.apply_controlled(Z, 0, 1)
    np.testing.assert_allclose(state.qarray, expected)

@pytest.mark.parametrize("bad_control, bad_target", [(0,0),(0,2),(-1,1),(0,1.5)])
def test_bad_values_cnot(bad_control, bad_target):
    q = QuantumState("10")
    with pytest.raises(ValueError):
        q.cnot(bad_control, bad_target)

def test_bad_values_controlled():
    q = QuantumState("10")
    with pytest.raises(TypeError):
        q.apply_controlled("X", 0, 1)

@pytest.mark.parametrize("bits", ["00","01","10","11"])
def test_hadamard_sandwich(bits):
    q1 = QuantumState(bits)
    q2 = QuantumState(bits)
    q1.apply(H, 0)
    q1.apply(H, 1)
    q1.cnot(0,1)
    q1.apply(H, 1)
    q1.apply(H, 0)
    q2.cnot(1,0)
    np.testing.assert_allclose(q1.qarray, q2.qarray, atol=1e-12)

@pytest.mark.parametrize("bits, expected", [("000",{"000":1000}),("001",{"001":1000}),("101",{"101":1000}),("111",{"111":1000}),])
def test_basic_sample(bits, expected):
    q = QuantumState(bits)
    assert (q.sample(1000) == expected)

@pytest.mark.parametrize("bits", ["000","001","010","100"])
def test_basic_measure(bits):
    q = QuantumState(bits)
    assert q.measure() == bits
    np.testing.assert_allclose(q.qarray, QuantumState(bits).qarray)

def test_sample_counts_sum_to_shots():
    q = QuantumState("000")
    q.apply(H, 0)
    q.apply(H, 1)
    q.apply(H, 2)
    counts = q.sample(1234)
    assert sum(counts.values()) == 1234
    assert all(len(k) == 3 for k in counts)

def test_sample_does_not_change_state():
    q = QuantumState("000")
    q.apply(H, 0)
    q.apply(H, 1)
    q.apply(H, 2)
    q_copy = q.qarray.copy()
    q.sample(1000)
    np.testing.assert_allclose(q.qarray, q_copy)

def test_sample_same_seed():
    q1 = QuantumState("000")
    q2 = QuantumState("000")
    q1.apply(H,0)
    q1.apply(H,2)
    q2.apply(H,0)
    q2.apply(H,2)
    assert (q1.sample(500, rng=np.random.default_rng(10)) == q2.sample(500, rng=np.random.default_rng(10)))

def test_measure_collapses_to_outcome():
    q = QuantumState("101")
    q.apply(H,0)
    q.apply(H,1)
    b = q.measure()
    np.testing.assert_allclose(q.qarray, np.eye(2**q.num_qubits)[int(b,2)])

def test_measure_twice_same():
    q = QuantumState("101")
    q.apply(H,0)
    q.apply(H,1)
    b = q.measure()
    assert q.measure() == b

def test_bell_sample():
    q = QuantumState("00")
    q.apply(H, 0)
    q.cnot(0,1)
    b = q.sample(10000, rng=np.random.default_rng(10))
    assert b.keys() == {"00", "11"}
    assert 4850 < b["00"] < 5150

def test_bell_measure_correlated():
    for _ in range(100):
        q = QuantumState("00")
        q.apply(H, 0)
        q.cnot(0,1)
        assert q.measure() in {"00", "11"}

def test_uniform_three_qubit():
    q = QuantumState("101")
    q.apply(H,0)
    q.apply(H,1)
    q.apply(H,2)
    b = q.sample(8000, rng=np.random.default_rng(10))
    assert len(b.keys()) == 8
    for val in b.values():
        assert 900 < val < 1100

@pytest.mark.parametrize("bad_shots", [0, -5, 2.5, True, "100", None])
def test_sample_rejects_invalid_shots(bad_shots):
    q = QuantumState("00")
    with pytest.raises(ValueError):
        q.sample(bad_shots)

def test_sample_accepts_numpy_int_shots():
    q = QuantumState("00")
    assert q.sample(np.int64(100)) == {"00": 100}