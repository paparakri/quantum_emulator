import numpy as np
from quantum_emulator.gates import *

class Qubit:
	G_rng = np.random.default_rng(None)

	def __init__(self, init_val):
		if type(init_val) != int and not isinstance(init_val, np.ndarray):
			raise ValueError("Init Val not an int or np.array")
		elif isinstance(init_val, np.ndarray):
			state = np.array(init_val, dtype=np.complex128)
			if state.shape != (2,):
				raise ValueError(f"State must have shape (2,), got {state.shape}")
			if not np.isclose(np.linalg.norm(state), 1.0):
				raise ValueError(f"State must be normalized, got norm {np.linalg.norm(state)}")
			self.qarray = state
		elif init_val==0:
			self.qarray = np.array([1,0], dtype=np.complex128)
		elif init_val==1:
			self.qarray = np.array([0,1], dtype=np.complex128)
		else:
			raise ValueError("Int not 0 or 1")

	def apply(self, gate:Gate):
		if not isinstance(gate, Gate):
			raise TypeError("Didn't pass a gate")
		self.qarray = gate.matrix@self.qarray

	def probabilities(self):
		return np.abs(self.qarray) ** 2

	def sample(self, shots=1000, rng = G_rng):
		if not isinstance(shots, int) or shots<=0:
			raise ValueError("shots must be a positive integer")
		probs = self.probabilities()
		u = rng.random(shots)
		zeros = np.count_nonzero(u < probs[0])
		return {"0": int(zeros), "1": int(shots-zeros)}

	def measure(self, rng = G_rng):
		probs = self.probabilities()
		if rng.random() < probs[0]:
			self.qarray = np.array([1,0], dtype=np.complex128)
			return 0
		else:
			self.qarray = np.array([0,1], dtype=np.complex128)
			return 1

PLUS = (np.array([1,1], dtype=np.complex128)/np.sqrt(2))
PLUS.flags.writeable = False
MINUS = (np.array([1,-1], dtype=np.complex128)/np.sqrt(2))
MINUS.flags.writeable = False