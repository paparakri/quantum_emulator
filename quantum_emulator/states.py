import numpy as np
from quantum_emulator.gates import Gate, X
from functools import reduce

class QuantumState:
	G_rng = np.random.default_rng(None)

	def __init__(self, array):
		if isinstance(array, str):
			if len(array) == 0:
				raise ValueError("Bitstring must not be empty")
			if not set(array) <= {"0", "1"}:
				raise ValueError(f"Bitstring may only contain '0' and '1', got '{array}'")

			self.num_qubits = len(array)
			state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
			state[int(array, 2)] = 1.0
			self.qarray = state

		elif isinstance(array, (np.ndarray, list, tuple)):
			state = np.array(array, dtype=np.complex128)
			if state.ndim != 1 or state.size < 2:
				raise ValueError(f"State must be a 1D array of length >= 2, got shape {state.shape}")
			length = state.size
			if length & (length - 1) != 0:
				raise ValueError(f"State length must be a power of 2, got {length}")
			if not np.isclose(np.linalg.norm(state), 1.0):
				raise ValueError(f"State must be normalized, got norm {np.linalg.norm(state)}")

			self.num_qubits = length.bit_length() - 1
			self.qarray = state
		else:
			raise TypeError(f"Expected a bitstring or array, got {type(array).__name__}")

	def probabilities(self):
		return np.abs(self.qarray) ** 2

	def apply(self, gate, target):
		if not isinstance(gate, Gate):
			raise TypeError("gate is not of type Gate")
		if gate.num_qubits != 1:
			raise ValueError("Multi-Qubit gates not ready")
		self._check_qubit_index(target, "target")
		temp_array = [np.eye(2, dtype=np.complex128)] * self.num_qubits
		temp_array[target] = gate.matrix
		gate_array = reduce(np.kron, temp_array)
		self.qarray = gate_array @ self.qarray

	def cnot(self, control, target):
		self.apply_controlled(X(), control, target)

	def apply_controlled(self, gate, control, target):
		if not isinstance(gate, Gate):
			raise TypeError("gate is not of type Gate")
		if gate.num_qubits != 1:
			raise ValueError("gate is not a 1-qubit Gate")
		self._check_qubit_index(target, "target")
		self._check_qubit_index(control, "control")
		if control == target:
			raise ValueError("control is equal to target")

		factors_a = [np.eye(2, dtype=np.complex128)] * self.num_qubits
		factors_a[control] = P0
		term_a = reduce(np.kron, factors_a)

		factors_b = [np.eye(2, dtype=np.complex128)] * self.num_qubits
		factors_b[control] = P1
		factors_b[target] = gate.matrix
		term_b = reduce(np.kron, factors_b)

		self.qarray = (term_a + term_b) @ self.qarray

	def measure(self, rng = G_rng):
		probs = self.probabilities()
		choice = rng.choice(2**self.num_qubits, p=probs)
		self.qarray.fill(0)
		self.qarray[choice] = 1
		return self._to_bitstring(choice)

	def sample(self, shots, rng=G_rng):
		if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool) or shots<=0:
			raise ValueError("shots must be a positive integer")
		probs = self.probabilities()
		u = rng.multinomial(shots, probs)
		return {
			self._to_bitstring(i): int(c)
			for i, c in enumerate(u)
			if c > 0
		}

	def _check_qubit_index(self, index, name):
		if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
			raise ValueError(f"{name} must be an integer")
		if not 0 <= index < self.num_qubits:
			raise ValueError(f"{name} index not in correct range")

	def _to_bitstring(self, index):
		return format(index, f"0{self.num_qubits}b")

PLUS = (np.array([1,1], dtype=np.complex128)/np.sqrt(2))
PLUS.flags.writeable = False
MINUS = (np.array([1,-1], dtype=np.complex128)/np.sqrt(2))
MINUS.flags.writeable = False

P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
P0.flags.writeable = False
P1.flags.writeable = False