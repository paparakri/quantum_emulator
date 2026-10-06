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


def test_hzh_equals_x():
    np.testing.assert_allclose(H().matrix @ Z().matrix @ H().matrix, X().matrix, atol=1e-12)


def test_apply_rejects_non_gate():
    q = Qubit(0)
    with pytest.raises(TypeError):
        q.apply("X")
