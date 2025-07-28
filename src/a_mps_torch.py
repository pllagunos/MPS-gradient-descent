"""Toy code implementing a matrix product state, ported to PyTorch
This version is designed for gradient-based optimization
"""

import torch
from typing import List
from scipy.linalg import svd


class MPS:
    """Class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    THe MPS is stored in a right-canonical form using the Vidal notation (B, S)

    Attributes
    ----------
    Bs : list of torch.Tensor
        The 'B' tensors in right-canonical form, one for each physical site.
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``
    Ss : list of torch.Tensor
        The Schmidt values (singular values) at each bond. `Ss[i]` is for the bond
        to the left of site `i`.
    L : int
        Number of sites.
    """

    def __init__(self, Bs: List[torch.Tensor], Ss: List[torch.Tensor]):
        self.Bs = Bs
        self.Ss = Ss
        self.L = len(Bs)

    def copy(self):
        new_Bs = [B.clone().detach() for B in self.Bs]
        new_Ss = [S.clone().detach() for S in self.Ss]
        return MPS(new_Bs, new_Ss)
    
    def get_tensors_as_list(self) -> List[torch.Tensor]:
        """Returns a flat list of all tensors in the MPS that have gradients."""
        tensors = []
        for B in self.Bs:
            if B.requires_grad:
                tensors.append(B)
        for S in self.Ss:
            if S.requires_grad:
                tensors.append(S)
        return tensors

    def get_theta1(self, i: int) -> torch.Tensor:
        """
        Calculate the effective single-site wave function on site `i`
        in mixed canonical form.

        The returned array has legs (vL, i, vR).
        """
        # vL [vL'], [vL] i vR -> vL i vR
        return torch.tensordot(torch.diag(self.Ss[i]), self.Bs[i], dims=([1], [0]))

    def get_theta2(self, i: int) -> torch.Tensor:
        """
        Calculate the effective two-site wave function on sites i, j=(i+1)
        in mixed canonical form.

        The returned array has legs (vL, i, j, vR).
        """
        j = i + 1
        # vL i [vR], [vL] j vR -> vL i j vR
        theta1 = self.get_theta1(i)
        return torch.tensordot(theta1, self.Bs[j], dims=([2], [0]))

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.L - 1)]

    def site_expectation_value(self, op: torch.Tensor) -> List[torch.Tensor]:
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            # op acts on physical leg 'i'
            op_theta = torch.tensordot(op, theta, dims=([1], [1]))  # i [i*], vL [i] vR -> i vL vR
            # contract with conjugate to get expectation value
            # [vL*] [i*] [vR*], [i] [vL] [vR]
            exp_val = torch.tensordot(theta.conj(), op_theta, dims=([0, 1, 2], [1, 0, 2]))
            result.append(exp_val.real)
        return result

    def bond_expectation_value(self, op_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Calculate expectation values of two-site operators on each bond."""
        result = []
        for i in range(self.L - 1):
            theta = self.get_theta2(i)  # vL i j vR
            # op acts on physical legs 'i' and 'j'
            op_theta = torch.tensordot(op_list[i], theta, dims=([2, 3], [1, 2])) # i j [i*] [j*], vL [i] [j] vR -> i j vL vR
            # contract with conjugate
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
            exp_val = torch.tensordot(theta.conj(), op_theta, dims=([0, 1, 2, 3], [2, 0, 1, 3]))
            result.append(exp_val.real)
        return result
    
    def norm_squared(self) -> torch.Tensor:
        """
        Calculates the squared norm <psi|psi> of the MPS.
        This is done by contracting the network from left to right.
        """
        # Start with a 1x1 identity matrix, representing the left boundary
        contr = torch.eye(1, dtype=self.Bs[0].dtype)
        for i in range(self.L):
            B = self.Bs[i] # vL, i, vR
            # Contract the current accumulated tensor with the MPS tensor B
            # contr: [vL*], vL
            # B: vL, i, vR
            temp = torch.tensordot(contr, B, dims=([1], [0])) # [vL*] [vL], i, vR -> [vL*] i vR
            # Contract with the conjugate of B to form the transfer matrix for the site
            # temp: [vL*] i vR
            # B.conj(): vL*, i*, vR*
            contr = torch.tensordot(temp, B.conj(), dims=([0, 1], [0, 1])) # [vL*] [i] vR, [vL*] [i*] vR* -> vR vR*
        # The final result should be a 1x1 matrix, its single element is the norm squared
        return contr.squeeze()
    
    def expectation_value_mpo(self, mpo: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the expectation value <psi|H_mpo|psi>.
        This is done by contracting the "sandwich" network from left to right.
        """
        # Start with the left boundary vector for the MPO contraction
        contr = torch.zeros(1, mpo[0].shape[0], 1, dtype=self.Bs[0].dtype)
        contr[0, 0, 0] = 1.0 # Corresponds to the identity operator at the start of the MPO

        for i in range(self.L):
            B_ket = self.Bs[i]       # vL, i, vR
            B_bra = B_ket.conj()   # vL*, i*, vR*
            W = mpo[i]             # wL, wR, i, i*

            # Contract with the ket
            # contr: vL_bra, wL, vL_ket
            # B_ket: vL_ket, i, vR_ket
            temp = torch.tensordot(contr, B_ket, dims=([2], [0])) # vL_bra, wL, [vL_ket], i, vR_ket -> vL_bra, wL, i, vR_ket
            # Contract with the MPO tensor W
            # temp: vL_bra, wL, i, vR_ket
            # W: wL, wR, i, i*
            temp = torch.tensordot(temp, W, dims=([1, 2], [0, 2])) # vL_bra, [wL], [i], vR_ket, wR, i* -> vL_bra, vR_ket, wR, i*
            # Contract with the bra
            # temp: vL_bra, vR_ket, wR, i*
            # B_bra: vL_bra, i*, vR_bra
            contr = torch.tensordot(temp, B_bra, dims=([0, 3], [0, 1])) # [vL_bra], vR_ket, wR, [i*], [vL_bra], [i*], vR_bra -> vR_ket, wR, vR_bra
            # Transpose to get the correct order for the next iteration: vR_bra, wR, vR_ket
            contr = contr.permute(2, 1, 0)
        
        # The final result is the element corresponding to the end of the MPO
        return contr[0, -1, 0].real
    
    def entanglement_entropy(self) -> List[float]:
        """Return the (von-Neumann) entanglement entropy for each bond."""
        result = []
        for i in range(1, self.L):
            S = self.Ss[i].clone()
            S = S[S > 1e-30]  # Avoid log(0)
            S2 = S * S
            # The norm of Schmidt values should be 1.
            assert abs(torch.linalg.norm(S) - 1.) < 1.e-10
            entropy = -torch.sum(S2 * torch.log(S2))
            result.append(entropy.item())
        return result


def init_spinup_MPS(L: int) -> MPS:
    """Return a product state with all spins up as a PyTorch MPS."""
    B = torch.zeros((1, 2, 1), dtype=torch.float64)
    B[0, 0, 0] = 1.
    S = torch.ones(1, dtype=torch.float64)
    Bs = [B.clone() for _ in range(L)]
    Ss = [S.clone() for _ in range(L + 1)] # L+1 bonds for L sites
    return MPS(Bs, Ss)

def init_random_mps(L: int, chi_max: int, d: int = 2) -> MPS:
    """Initializes a random MPS with requires_grad=True for optimization."""
    Bs = []
    Ss = []
    
    # Left-most bond dimension is 1
    chi_left = 1
    
    # First S matrix (bond 0)
    s0 = torch.ones(1, dtype=torch.float64, requires_grad=True)
    Ss.append(s0)

    for i in range(L):
        chi_right = min(chi_max, d ** (i + 1), d ** (L - i - 1))
        if i == L - 1:
            chi_right = 1

        # Random B tensor
        B = torch.randn((chi_left, d, chi_right), dtype=torch.float64, requires_grad=True)
        Bs.append(B)
        
        # Random S matrix
        s = torch.rand(chi_right, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            s = s / torch.linalg.norm(s)
        Ss.append(s)
        
        chi_left = chi_right
        
    return MPS(Bs, Ss)

def split_truncate_theta(theta: torch.Tensor, chi_max: int, eps: float):
	"""
	Split and truncate a two-site wave function in mixed canonical form.

	Split a two-site wave function as follows::
		vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
				|   |                       |             |
				i   j                       i             j

	Afterwards, truncate in the new leg (labeled ``vC``).

	Parameters
	----------
	theta : torch.Tensor
		Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
	chi_max : int
		Maximum number of singular values to keep
	eps : float
		Discard any singular values smaller than that.

	Returns
	-------
	A : torch.Tensor
		Left-canonical matrix on site i, with legs ``vL, i, vC``
	S : torch.Tensor
		Singular/Schmidt values.
	B : torch.Tensor
		Right-canonical matrix on site j, with legs ``vC, j, vR``
	"""
	chivL, dL, dR, chivR = theta.shape
	theta = theta.reshape(chivL * dL, dR * chivR)

	# SVD
	U, S, Vh = torch.linalg.svd(theta, full_matrices=False)

	# Truncate
	chivC = min(chi_max, torch.sum(S > eps).item())
	assert chivC >= 1
	# Keep the largest `chivC` singular values   
	piv = torch.argsort(S, descending=True)[:chivC]
	U, S, Vh = U[:, piv], S[piv], Vh[piv, :]

	# Renormalize
	S = S / torch.linalg.norm(S)

	# Reshape back to tensors
	A = U.reshape(chivL, dL, chivC)
	B = Vh.reshape(chivC, dR, chivR)

	return A, S, B