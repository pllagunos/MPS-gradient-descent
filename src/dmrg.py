"""
Toy code implementing the density-matrix renormalization group (DMRG).
This version is a faithful port of the original NumPy/SciPy implementation,
adapted to work with a PyTorch MPS object for comparison purposes.
"""

import torch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg._eigen.arpack as arp
from src.a_mps_torch import MPS, split_truncate_theta
from src.b_model_torch import TFIModel
from typing import List

class HEffective(scipy.sparse.linalg.LinearOperator):
    """
    Class for the effective Hamiltonian, directly adapted from the original NumPy version.
    It defines the matrix-vector product for use with scipy's iterative eigensolvers.
    """
    def __init__(self, LP: np.ndarray, RP: np.ndarray, W1: np.ndarray, W2: np.ndarray):
        self.LP = LP  # vL wL vL*
        self.RP = RP  # vR* wR vR
        self.W1 = W1  # wL wC i i*
        self.W2 = W2  # wC wR j j*
        chi_left, _, _ = LP.shape
        chi_right, _, _ = RP.shape
        _, _, d, _ = W1.shape
        self.theta_shape = (chi_left, d, d, chi_right)
        self.shape = (chi_left * d * d * chi_right, chi_left * d * d * chi_right)
        self.dtype = W1.dtype

    def _matvec(self, theta_flat: np.ndarray) -> np.ndarray:
        """Calculates the matrix-vector product H_eff * theta using NumPy."""
        theta = theta_flat.reshape(self.theta_shape)
        x = np.tensordot(self.LP, theta, axes=([2], [0]))
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))
        x = np.tensordot(x, self.W2, axes=([3, 1], [0, 3]))
        x = np.tensordot(x, self.RP, axes=([1, 3], [2, 1]))
        return x.reshape(-1)

class DMRGEngine:
    """
    DMRG algorithm adapted to operate on a PyTorch MPS object.
    The internal calculations are performed in NumPy for stability.
    """
    def __init__(self, psi: MPS, model: TFIModel, chi_max: int = 100):
        self.psi = psi
        self.model = model
        self.chi_max = chi_max
        self.LPs = [None] * self.psi.L
        self.RPs = [None] * self.psi.L
        self._initialize_environments()

    def _initialize_environments(self):
        """Pre-calculates the right environments before the first sweep."""
        H_mpo_np = [w.cpu().numpy() for w in self.model.H_mpo]
        D = H_mpo_np[-1].shape[1]
        chi = self.psi.Bs[-1].shape[2]
        RP = np.zeros((chi, D, chi), dtype=np.float64)
        RP[0, D - 1, 0] = 1.0
        self.RPs[-1] = RP
        for i in range(self.psi.L - 1, 0, -1):
            self._update_RP(i)

    def sweep(self):
        """Performs a single DMRG sweep (left-to-right and right-to-left)."""
        for i in range(self.psi.L - 1):
            self._update_bond(i)
        for i in range(self.psi.L - 2, -1, -1):
            self._update_bond(i)

    def _update_bond(self, i: int):
        j = i + 1
        
        if self.LPs[i] is None:
            D = self.model.H_mpo[0].cpu().numpy().shape[0]
            chi = self.psi.Bs[i].detach().cpu().numpy().shape[0]
            LP = np.zeros((chi, D, chi), dtype=np.float64)
            LP[0, 0, 0] = 1.0
            self.LPs[i] = LP

        Heff = HEffective(self.LPs[i], self.RPs[j], self.model.H_mpo[i].cpu().numpy(), self.model.H_mpo[j].cpu().numpy())
        
        theta0_torch = self.psi.get_theta2(i)
        theta0_np = theta0_torch.detach().cpu().numpy().reshape(-1)
        
        e, v = arp.eigsh(Heff, k=1, which='SA', v0=theta0_np.astype(Heff.dtype), return_eigenvectors=True)
        
        theta_np = v[:, 0].reshape(Heff.theta_shape)
        
        theta_torch = torch.from_numpy(theta_np)
        Ai, Sj, Bj = split_truncate_theta(theta_torch, self.chi_max, eps=1.e-14)
        
        # Update MPS tensors using the logic from the original TEBD code
        # This ensures the MPS remains in a manageable (though not strictly canonical) form
        Ss_i_inv_np = np.diag(1. / self.psi.Ss[i].detach().cpu().numpy())
        Gi_np = np.tensordot(Ss_i_inv_np, Ai.cpu().numpy(), axes=([1], [0]))
        
        self.psi.Bs[i] = torch.from_numpy(np.tensordot(Gi_np, np.diag(Sj.cpu().numpy()), axes=([2], [0])))
        self.psi.Ss[j] = Sj
        self.psi.Bs[j] = Bj
        
        self._update_LP(i)
        self._update_RP(j)

    def _update_RP(self, i: int):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        j = i - 1
        B = self.psi.Bs[i].detach().cpu().numpy()
        W = self.model.H_mpo[i].cpu().numpy()
        RP_i = self.RPs[i]
        
        temp = np.tensordot(B, RP_i, axes=([2], [0]))
        temp = np.tensordot(temp, W, axes=([1, 2], [3, 1]))
        RP_j = np.tensordot(temp, B.conj(), axes=([1, 3], [2, 1]))
        self.RPs[j] = RP_j

    def _update_LP(self, i: int):
        """Calculate LP left of site `i+1` from LP left of site `i`."""
        j = i + 1
        
        # This function requires a left-canonical 'A' tensor.
        # We calculate it on the fly from the 'B' and 'S' tensors.
        B_i_np = self.psi.Bs[i].detach().cpu().numpy()
        Ss_i_np = self.psi.Ss[i].detach().cpu().numpy()
        Ss_j_np = self.psi.Ss[j].detach().cpu().numpy()
        
        G = np.tensordot(B_i_np, np.diag(1./Ss_j_np), axes=([2],[0]))
        A = np.tensordot(np.diag(Ss_i_np), G, axes=([1],[0]))
        Ac = A.conj()
        
        W = self.model.H_mpo[i].cpu().numpy()
        LP_i = self.LPs[i]
        
        temp = np.tensordot(LP_i, A, axes=([2], [0]))
        temp = np.tensordot(W, temp, axes=([0, 3], [1, 2]))
        LP_j = np.tensordot(Ac, temp, axes=([0, 1], [2, 1]))
        self.LPs[j] = LP_j
