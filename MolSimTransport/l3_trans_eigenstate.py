import os
import numpy as np
import scipy.linalg
import MolSimTransport.utils.find_Gr_and_Gamma as findG
from MolSimTransport.utils.EC_molden import generate_EC_molden

def main():
    # Specify the target energy point
    try:
        input_energy = float(input(">>> Enter the energy point for calculating the eigenstate (float): "))
    except ValueError:
        print("Invalid input. Please enter a valid floating-point number.")
        input_energy = None

    with open("mpsh_eigenvalues.txt", "r") as f:
        lines = f.readlines()
        offset_efermi = float(lines[1].strip())
        mpsh_eigenenergy = np.array([float(line.strip()) for line in lines[3:]])

    with open("mpsh_eigenvectors.txt", "r") as f:
        lines = f.readlines()[1:]
        mpsh_eigenvector = np.loadtxt(lines)

    target_energy = input_energy - offset_efermi

    s_file_name = next((f for f in os.listdir() if f.startswith('device_s') and f.endswith('.dat')), None)
    if not s_file_name:
        raise FileNotFoundError("No file found starting with 'device_s' and ending with '.dat'")
    s = np.loadtxt(s_file_name, skiprows=5)

    # Choosing electrode type
    elec_type = input(">>> Choose electrode type (s / l): ").strip().lower()

    if elec_type == "l":
        npl = 972
    elif elec_type == "s":
        npl = 324
    else:
        raise ValueError("Invalid input! Please enter 's' or 'l'.")

    # npl = 972
    nc = len(s)
    nm = nc - 2 * npl
    s_m = s[npl:npl+nm, npl:npl+nm]

    # Load Gr, GammaL, and GammaR that are closest to the target energy point within the tolerance range
    Gr, Gamma_L, Gamma_R, closest_energy = findG.find_closest_G_and_Gamma(target_energy, offset_efermi)

    closest_energy_str = f"{closest_energy:.5f}"

    AL = np.dot(Gr, np.dot(Gamma_L, np.conj(Gr.transpose())))

    S_sqr2 = scipy.linalg.sqrtm(s_m)
    S_invsqr2 = np.linalg.inv(S_sqr2)

    # Calculate the transmission eigenstate according to the method in "PRB, 76, 115117, 2007"
    ABar_L = S_sqr2 @ AL @ S_sqr2
    lamb, U = np.linalg.eig(ABar_L)
    Utilde = U @ np.diagflat(np.sqrt(1. / (2. * np.pi) * lamb))
    GammaBar_R = S_invsqr2 @ Gamma_R @ S_invsqr2
    TM = 2. * np.pi * Utilde.conj().T @ GammaBar_R @ Utilde
    T, c = np.linalg.eig(TM)
    Pd = S_invsqr2 @ Utilde @ c

    if input_energy is not None:
        filename = f"trans_eigenvalues_{closest_energy_str}.txt"
        np.savetxt(filename, np.real(T), fmt='%22.16f')
    
    trans_eigenvector = Pd[:,0] # Calculate only the eigenvector corresponding to the first (largest) eigenvalue.

    # if input_energy is not None:
    #     filename = f"first_trans_eigenvector_{closest_energy_str}.txt"
    #     np.savetxt(filename, trans_eigenvector, fmt='%22.16f')

    # Get the norm of the projected eigenstate
    norm_ev = trans_eigenvector.conj().T @ s_m @ trans_eigenvector

    print(" ")
    print("The MPSH orbitals with projection weight greater than 0.001 in this eigenstate:")
    print(" ")
    # Project the eigenstate onto the extended molecule
    # Loop over all MPSH
    for i in range(len(mpsh_eigenenergy )):
        # Resolve the eigenstate into the MPSH states
        coeff = mpsh_eigenvector[:, i].conj().T @ s_m @ trans_eigenvector
        # Find the strength of the MPSH projection
        p = np.conj(coeff)*coeff
        p = np.abs(p/norm_ev)
        # Print out the weight of each non negligible projection
        if p > 0.001:
            print(i+1, p)
    print(" ")
    generate_EC_molden(closest_energy, trans_eigenvector) # Generate the Molden file for the EigenChannel.

if __name__ == "__main__":
    main()
