import numpy as np
import pytest

from quantum_emulator.states import Qubit, PLUS, MINUS
from quantum_emulator.gates import *

@pytest.mark.parametrize("init_val, result", [(0, [0, 1]), (1, [1, 0]),])
def test_basis_state_X_gate(init_val, result):
    q1 = Qubit(init_val)
    q1.apply(X())
    np.testing.assert_allclose(q1.qarray, result)

@pytest.mark.parametrize("init_val, result", [(PLUS,MINUS), (MINUS,PLUS)])
def test_basis_state_Z_gate(init_val, result):
    q1 = Qubit(init_val)
    q1.apply(Z())
    np.testing.assert_allclose(q1.qarray, result)

@pytest.mark.parametrize("init_val, result", [(0,PLUS), (1,MINUS)])
def test_basis_state_H_gate(init_val, result):
    q1 = Qubit(init_val)
    q1.apply(H())
    np.testing.assert_allclose(q1.qarray, result)

class NonSymmetric(Gate):
    # Unitary but not symmetric, so M @ v and v @ M give different results.
    name = "NonSymmetric"
    matrix = np.array([[0, 1], [-1, 0]], dtype=np.complex128)


def test_apply_multiplies_matrix_on_the_left():
    # M|0> is the first column of M: [0, -1]. The wrong order would give the first row: [0, 1].
    q = Qubit(0)
    q.apply(NonSymmetric())
    np.testing.assert_allclose(q.qarray, [0, -1])


@pytest.mark.parametrize("init_val, result", [(0, [1, 0]), (1, [0, -1])])
def test_z_gate_on_computational_basis(init_val, result):
    q = Qubit(init_val)
    q.apply(Z())
    np.testing.assert_allclose(q.qarray, result)


@pytest.mark.parametrize("gate", [X(), Z(), H()])
def test_gate_is_its_own_inverse(gate):
    np.testing.assert_allclose(gate.matrix @ gate.matrix, np.eye(2), atol=1e-12)

@pytest.mark.parametrize("gate", [X(), Z(), H(), NonSymmetric()])
def test_gate_is_unitary(gate):
    assert is_unitary(gate.matrix)

@pytest.mark.parametrize("gate_matrix", [np.array([[1,1],[0,1]], dtype=np.complex128), np.array([[1,1,0],[0,0,1]], dtype=np.complex128), np.array([1,1], dtype=np.complex128)])
def test_gate_is_not_unitary(gate_matrix):
    assert not is_unitary(gate_matrix)

@pytest.mark.parametrize("gate_matrix", [np.array([[1,1],[0,1]], dtype=np.complex128), np.array([[1,1,0],[0,0,1]], dtype=np.complex128), np.array([1,1], dtype=np.complex128)])
def test_creating_non_unitary_gate(gate_matrix):
    with pytest.raises(ValueError):
        class BadGate(Gate):
            name = "BadGate"
            matrix = gate_matrix

def test_is_unitary_respects_tolerance():
    # Hand-typed 0.7071 instead of 1/sqrt(2): diagonal of U†U is off by ~1.9e-5.
    rounded_h = np.array([[0.7071, 0.7071], [0.7071, -0.7071]], dtype=np.complex128)
    assert not is_unitary(rounded_h)
    assert is_unitary(rounded_h, atol=1e-4)

def test_hzh_equals_x():
    np.testing.assert_allclose(H().matrix @ Z().matrix @ H().matrix, X().matrix, atol=1e-12)


def test_apply_rejects_non_gate():
    q = Qubit(0)
    with pytest.raises(TypeError):
        q.apply("X")

@pytest.mark.parametrize("init_val, result", [
        (0, np.array([1,0], dtype=np.float64)),
        (1, np.array([0,1], dtype=np.float64)),
        (PLUS, np.array([0.5,0.5], dtype=np.float64)),
        (MINUS, np.array([0.5,0.5], dtype=np.float64)),
        (np.array([1,1j])/np.sqrt(2), np.array([0.5,0.5], dtype=np.float64)),
    ])
def test_probabilities(init_val, result):
    q1 = Qubit(init_val)
    probabilities = q1.probabilities()
    assert probabilities.shape == (2,)
    assert probabilities.dtype == np.float64
    p_sum = sum(probabilities)
    np.testing.assert_allclose(p_sum, 1)
    np.testing.assert_allclose(probabilities, result)

def test_Z_probabilities():
    q1=Qubit(PLUS)
    q2=Qubit(PLUS)
    q2.apply(Z())
    np.testing.assert_allclose(q1.probabilities(), q2.probabilities())