"""
Toy code implementing the time evolving block decimation (TEBD), ported to PyTorch.
"""

import torch
from src.a_mps_torch import MPS, split_truncate_theta
from src.b_model_torch import TFIModel
from typing import List

def calc_U_bonds(model: TFIModel, dt: float) -> List[torch.Tensor]:
    """
    Given a model, calculate U_bonds[i] = expm(-dt*model.H_bonds[i]).

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in).
    Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        # Reshape H to a matrix to exponentiate
        H_matrix = H.reshape(d * d, d * d)
        # Use torch.matrix_exp for the matrix exponential
        U = torch.matrix_exp(-dt * H_matrix)
        U_bonds.append(U.reshape(d, d, d, d))
    return U_bonds


def run_TEBD(psi: MPS, U_bonds: List[torch.Tensor], N_steps: int, chi_max: int, eps: float):
    """
    Evolve the state `psi` for `N_steps` time steps with (first order) TEBD.
    The state psi is modified in place.
    """
    Nbonds = psi.L - 1
    assert len(U_bonds) == Nbonds
    for n in range(N_steps):
        # Apply gates to even bonds, then odd bonds
        for k in [0, 1]:  # 0 for even, 1 for odd
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)


def update_bond(psi: MPS, i: int, U_bond: torch.Tensor, chi_max: int, eps: float):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = i + 1
    
    # 1. Construct the two-site wavefunction theta
    theta = psi.get_theta2(i)  # Legs: vL, i, j, vR
    
    # 2. Apply the two-site gate U_bond
    # U_bond legs: i, j, i*, j*
    # theta legs: vL, i, j, vR
    # Contraction: U_bond[i,j,i*,j*] * theta[vL,i*,j*,vR]
    Utheta = torch.tensordot(U_bond, theta, dims=([2, 3], [1, 2]))  # Legs: i, j, vL, vR
    
    # Transpose to bring virtual legs to the outside
    Utheta = Utheta.permute(2, 0, 1, 3)  # Legs: vL, i, j, vR
    
    # 3. Split and truncate using SVD
    Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
    
    # 4. Put the new tensors back into the MPS
    # This step updates the MPS tensors while maintaining the canonical form structure.
    # We need to absorb the inverse of the old Schmidt values on the left
    # and multiply by the new ones on the right.
    Gi = torch.tensordot(torch.diag(1. / psi.Ss[i]), Ai, dims=([1], [0]))
    psi.Bs[i] = torch.tensordot(Gi, torch.diag(Sj), dims=([2], [0]))
    psi.Ss[j] = Sj
    psi.Bs[j] = Bj


def example_TEBD_gs_finite(L: int, J: float, g: float):
    """Example of finding the ground state using imaginary time evolution with TEBD."""
    print("finite TEBD (imaginary time evolution) with PyTorch")
    print(f"L={L}, J={J:.1f}, g={g:.2f}")
    
    import src.a_mps_torch
    
    model = TFIModel(L, J=J, g=g)
    psi = src.a_mps_torch.init_spinup_MPS(L)
    
    for dt in [0.1, 0.01, 0.001, 1.e-4, 1.e-5]:
        U_bonds = calc_U_bonds(model, dt)
        run_TEBD(psi, U_bonds, N_steps=100, chi_max=30, eps=1.e-12)
        E = model.energy(psi)
        print(f"dt = {dt:.5f}: E = {E.item():.13f}")
        
    print("Final bond dimensions: ", psi.get_chi())
    
    # For small systems, compare to exact diagonalization (requires porting tfi_exact or using original)
    # E_exact = ... 
    # print(f"Exact diagonalization: E = {E_exact:.13f}")
    # print(f"Relative error: {abs((E.item() - E_exact) / E_exact)}")
    
    return E, psi, model


if __name__ == "__main__":
    example_TEBD_gs_finite(L=14, J=1., g=1.5)
