"""
Toy code implementing the transverse-field ising model, ported to PyTorch.
"""

import torch
from typing import List
from src.a_mps_torch import MPS

class TFIModel:
	"""
	Class generating the Hamiltonian of the transverse-field Ising model using PyTorch.

	The Hamiltonian reads
	H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i

	Parameters
	----------
	L : int
		Number of sites.
	J, g : float
		Coupling parameters of the above defined Hamiltonian.
	device : str
		The device to store the tensors on, e.g., 'cpu' or 'cuda'.
	"""

	def __init__(self, L: int, J: float, g: float, device: str = 'cpu'):
		self.L, self.d = L, 2
		self.J, self.g = J, g
		self.device = device
		
		# Define Pauli matrices as torch tensors
		self.sigmax = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64, device=self.device)
		self.sigmay = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex128, device=self.device)
		self.sigmaz = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float64, device=self.device)
		self.id = torch.eye(2, dtype=torch.float64, device=self.device)
		
		self.H_bonds = self._init_H_bonds()
		self.H_mpo = self._init_H_mpo()

	def _init_H_bonds(self) -> List[torch.Tensor]:
		"""Initialize `H_bonds` hamiltonian. Called by __init__()."""
		sx, sz, id = self.sigmax, self.sigmaz, self.id
		d = self.d
		H_list = []
		
		for i in range(self.L - 1):
			gL = gR = 0.5 * self.g
			# For finite systems, the boundary terms are different
			if i == 0:
				gL = self.g
			if i == self.L - 2: # Corrected boundary condition for the last bond
				gR = self.g
			
			# Construct the two-site Hamiltonian term for the bond
			H_bond = -self.J * torch.kron(sx, sx) - gL * torch.kron(sz, id) - gR * torch.kron(id, sz)
			
			# H_bond has legs (i out, j out, i in, j in)
			H_list.append(H_bond.reshape(d, d, d, d))
			
		return H_list
	

	def _init_H_mpo(self) -> List[torch.Tensor]:
		"""Initialize `H_mpo` representation. Used for DMRG."""
		w_list = []
		for i in range(self.L):
			# MPO tensor for the bulk
			w = torch.zeros((3, 3, self.d, self.d), dtype=torch.float64, device=self.device)
			w[0, 0] = self.id
			w[2, 2] = self.id
			w[0, 1] = self.sigmax
			w[0, 2] = -self.g * self.sigmaz
			w[1, 2] = -self.J * self.sigmax
			#  W = np.array([[id, sx, -g*sz], [zeros, zeros, -J*sx], [zeros, zeros, id]])
			w_list.append(w)
				
		return w_list	
	
	def energy(self, psi: MPS) -> torch.Tensor:
		"""Evaluate energy E = <psi|H|psi> using the bond representation."""
		assert psi.L == self.L
		bond_energies = psi.bond_expectation_value(self.H_bonds)
		return torch.sum(torch.stack(bond_energies))
			
	def energy_mpo(self, psi: MPS) -> torch.Tensor:
			"""Evaluate energy E = <psi|H_mpo|psi> using the MPO representation."""
			assert psi.L == self.L
			return psi.expectation_value_mpo(self.H_mpo)

