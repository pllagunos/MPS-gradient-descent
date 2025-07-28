"""
Finds the ground state of the Transverse-Field Ising Model using
variational gradient descent on a Matrix Product State.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.a_mps_torch import init_random_mps
from src.b_model_torch import TFIModel

def run(psi, L: int, J: float, g: float, chi_max: int, num_steps: int, learning_rate: float):
    """
    Performs gradient descent to find the ground state of the TFI model.

    Parameters:
    -----------
    Psi : MPS
        Initial MPS state to optimize.
    L : int
        Number of sites in the chain.
    J, g : float
        Model parameters for the TFI Hamiltonian.
    chi_max : int
        The maximum bond dimension of the MPS.
    num_steps : int
        The number of optimization steps to perform.
    learning_rate : float
        The learning rate for the Adam optimizer.

    Returns:
    --------
    final_energy : float
        The final ground state energy found.
    energies : list
        A list of the energy at each optimization step.
    """
    print("Finding ground state with Gradient Descent...")
    print(f"L={L}, J={J}, g={g}, chi_max={chi_max}")
    print(f"num_steps={num_steps}, learning_rate={learning_rate}")

    # 1. Initialize a random MPS with requires_grad=True
    # This is the variational state we will optimize.
    psi = init_random_mps(L, chi_max=chi_max)
    
    # 2. Initialize the model Hamiltonian (MPO representation)
    model = TFIModel(L, J=J, g=g)

    # 3. Set up the optimizer
    # We use the Adam optimizer, which is generally robust.
    # We pass it all the tensors of our MPS that we want to train.
    optimizer = torch.optim.Adam(psi.get_tensors_as_list(), lr=learning_rate)

    energies = []

    # 4. The main optimization loop
    for step in range(num_steps):
        # --- Core Gradient Descent Step ---
        
        # a) Zero out gradients from the previous step
        optimizer.zero_grad()
        
        # b) Define the loss function: the normalized energy
        #    E = <psi|H|psi> / <psi|psi>
        energy = model.energy_mpo(psi)
        norm_sq = psi.norm_squared()
        loss = energy / norm_sq
        
        # c) Backpropagation: PyTorch automatically computes the
        #    gradient of the loss with respect to all MPS tensors.
        loss.backward()
        
        # d) Optimizer step: Update all MPS tensors by taking a
        #    small step in the direction opposite to the gradient.
        optimizer.step()
        
        # --- End of Core Step ---

        # Store and print energy for monitoring
        current_energy = loss.item()
        energies.append(current_energy)
        if step % 50 == 0 or step == num_steps - 1:
            print(f"Step {step:5d} | Energy = {current_energy:.12f}")
            
    final_energy = energies[-1]
    print(f"Gradient descent finished. Final energy: {final_energy:.12f}")
    return final_energy, energies

if __name__ == '__main__':
    # --- Parameters ---
    L = 14
    J = 1.0
    g = 1.5
    chi_max = 30
    
    # --- Run Gradient Descent ---
    gd_steps = 100
    gd_lr = 0.1
    final_E_gd, energies_gd = run(L, J, g, chi_max, gd_steps, gd_lr)

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(energies_gd, label='Gradient Descent')
    
    # Add theoretical ground state energy for comparison
    try:
        from tfi_exact import finite_gs_energy
        theoretical_energy = finite_gs_energy(L, J, g)
        plt.axhline(y=theoretical_energy, color='r', linestyle='--', label=f'Exact GS Energy ({theoretical_energy:.6f})')
    except ImportError:
        print("\nCould not import `tfi_exact`. Skipping theoretical energy plot.")

    plt.xlabel("Optimization Steps")
    plt.ylabel("Energy")
    plt.title(f"Ground State Search (L={L}, J={J}, g={g})")
    plt.legend()
    plt.grid(True)
    plt.show()
