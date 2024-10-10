'''
A python script that computes the transport of "extended molecule + cluster electrode" under the xTB method.
'''

from tblite.interface import Calculator
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import matplotlib
import time

def read_xyz(filename):
    atom_to_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                      'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
                      'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
                      'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
                      'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
                      'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
                      'Ba': 56, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
                      'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86}  # 1-56, 72-86

    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())

    numbers, positions, elements = [], [], []
    for line in lines[2:2+natoms]:
        parts = line.strip().split()
        atom = parts[0]
        coords = [float(x) / 0.52917721067 for x in parts[1:4]]  # Angstrom to Bohr
        numbers.append(atom_to_number[atom])
        positions.append(coords)
    return np.array(numbers), np.array(positions)

def calculate_transmission(xyz_filename, method, E_number, ClusterNum):
    start_time = time.time()
    numbers, positions = read_xyz(xyz_filename)

    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    res = calc.singlepoint()

    OrbE = 27.2114 * res.get("orbital-energies")
    S_full = res.get("overlap-matrix")
    C = np.transpose(res.get("orbital-coefficients"))
    E = np.diag(OrbE.flatten())

    H_cec = C @ E @ la.inv(C)
    H_full = S_full @ H_cec

    FermiEnergy25 = -11.50683  # Defined as the HOMO energy of 25 Au clusters
    FermiEnergy28 = -11.14726  # Defined as the HOMO energy of 28 Au clusters

    print(" ")
    EnergyRange = float(input(">>> Specify the energy range (from 'Fermi energy' plus or minus 'energy range', e.g., 4) (eV): "))

    while True:
        try:
            if ClusterNum == 25:
                EF = FermiEnergy25
                break
            elif ClusterNum == 28:
                EF = FermiEnergy28
                break
            else:
                print("Invalid input. Please enter an integer (25 or 28).")
        except ValueError:
            print("Invalid input. Please enter an integer (25 or 28).")

    print("Starting transport calculation...")

    E_num = E_number
    Erange = np.linspace(EF - EnergyRange, EF + EnergyRange, E_num)
    z_plus = 1e-13j

    N_L = ClusterNum * 9  # Define the range of the left and right Cluster ("25(28)Au" + "3(0)/4(1)Au + M + 3(0)/4(1)Au" + "25(28)Au")
    N_R = N_L
    N_m = len(H_full) - N_L - N_R  # extended molecule actually

    H_L = H_full[:N_L, :N_L]
    H_m = H_full[N_L:N_L+N_m, N_L:N_L+N_m]
    H_R = H_full[N_L+N_m:, N_L+N_m:]
    V_Lm = H_full[:N_L, N_L:N_L+N_m]
    V_mL = H_full[N_L:N_L+N_m, :N_L]
    V_Rm = H_full[N_L+N_m:, N_L:N_L+N_m]
    V_mR = H_full[N_L:N_L+N_m, N_L+N_m:]

    S_L = S_full[:N_L, :N_L]
    S_m = S_full[N_L:N_L+N_m, N_L:N_L+N_m]
    S_R = S_full[N_L+N_m:, N_L+N_m:]
    S_Lm = S_full[:N_L, N_L:N_L+N_m]
    S_mL = S_full[N_L:N_L+N_m, :N_L]
    S_Rm = S_full[N_L+N_m:, N_L:N_L+N_m]
    S_mR = S_full[N_L:N_L+N_m, N_L+N_m:]

    LDOS = 0.036  # eV-1
    g_L = -1j * np.pi * LDOS * np.eye(N_L)
    g_R = -1j * np.pi * LDOS * np.eye(N_R)

    Tpara = np.zeros(len(Erange))

    for i, EE in enumerate(Erange):
        Sigma_L = (EE * S_mL - V_mL) @ g_L @ (EE * S_Lm - V_Lm)
        Sigma_R = (EE * S_mR - V_mR) @ g_R @ (EE * S_Rm - V_Rm)
        Gamma_L = 1j * (Sigma_L - Sigma_L.T.conj())
        Gamma_R = 1j * (Sigma_R - Sigma_R.T.conj())

        G = la.inv((EE + z_plus) * S_m - H_m - Sigma_L - Sigma_R)
        Tpara[i] = np.real(np.trace(G @ Gamma_L @ G.T.conj() @ Gamma_R))

    ## Save transmission spectrum to txt file ##
    np.savetxt('Transmission.txt', np.column_stack((Erange, np.abs(Tpara))),
               fmt='%22.15f\t%22.15f', delimiter='\t', header='Erange\tTransmission', comments='')

    matplotlib.use('Agg')
    plt.figure()
    plt.plot(Erange, np.log10(np.abs(Tpara)), label='Transmission', color='blue')
    plt.title('Transmission Spectrum')
    plt.xlim([EF - EnergyRange, EF + EnergyRange])
    plt.xlabel('Energy (eV)', fontweight='bold')
    plt.ylabel('Transmission (log10)', fontweight='bold')
    plt.legend()
    plt.savefig('Transmission.png')

    print(" ")
    print(f"Chosen method:  {method}")
    print(f"\"Fermi energy\" (Defined as the HOMO energy of {ClusterNum} Au clusters) (eV): ", EF )
    print(" ")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print("Done!")

def main():
    xyz_filename = input(">>> Enter XYZ file name (e.g., coor.xyz): ")
    method_mapping = {1: "GFN1-xTB", 2: "GFN2-xTB"}
    chosen_method = method_mapping[int(input(">>> Enter calculation method (1 for GFN1-xTB, 2 for GFN2-xTB, e.g., 1): "))]
    E_number = int(input(">>> Specify the energy number, (e.g., 2001): "))
    ClusterNum = int(input(">>> Specify the cluster atom number (25 or 28) (Select 25 unless you want to use adatom in EM): "))
    calculate_transmission(xyz_filename, chosen_method, E_number, ClusterNum)

if __name__ == "__main__":
    main()
