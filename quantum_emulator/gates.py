import numpy as np

def is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix.conj().T @ matrix, identity, rtol=0, atol=atol)

class Gate:
    name: str
    matrix: np.ndarray

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.matrix.flags.writeable = False
        if not is_unitary(cls.matrix):
            raise ValueError(f"The gate {cls.name} is not unitary")

    @property
    def num_qubits(self) -> int:
        # A 2^k x 2^k matrix acts on k qubits.
        return int(np.log2(self.matrix.shape[0]))

    def __repr__(self) -> str:
        return f"{self.name}()"


class X(Gate):
    name = "X"
    matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)

class Z(Gate):
    name = "Z"
    matrix = np.array([[1,0],[0,-1]], dtype=np.complex128)

class H(Gate):
    name = "H"
    matrix = np.array([[1,1],[1,-1]], dtype=np.complex128)/np.sqrt(2)