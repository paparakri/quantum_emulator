import numpy as np


class Gate:
    """Base class for all gates. Subclasses only define `name` and `matrix`."""

    name: str
    matrix: np.ndarray

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.matrix.flags.writeable = False

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