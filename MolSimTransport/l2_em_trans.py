'''
A python script that computes the transport of "extended molecule + cluster electrode" under the xTB method.
'''

from tblite.interface import Calculator
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import matplotlib
import math
import json
from importlib.resources import files
from MolSimTransport.utils.functions import l2_calculate_transmission
import time

def read_xyz(filename):

    atom_to_number = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('atomic_num.json'), 'r'))

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

def calculate_transmission(xyz_filename, method):
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
    ClusterNum = int(input(">>> Specify the cluster atom number (25 or 28) (Select 25 unless you want to use adatom in EM): "))
    EnergyRange = float(input(">>> Specify the energy range (from 'Fermi energy' plus or minus 'energy range', e.g., 4) (eV): "))
    EnergyInterval = float(input(">>> Specify the energy interval (e.g., 0.01) (eV): "))

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

    E_num = math.ceil(EnergyRange * 2 / EnergyInterval)
    Erange = np.round(np.linspace(EF - EnergyRange, EF + EnergyRange, E_num+1), 5)
    z_plus=1e-13j

    N_L = ClusterNum * 9  # Define the range of the left and right Cluster ("25(28)Au" + "3(0)/4(1)Au + M + 3(0)/4(1)Au" + "25(28)Au")
    N_R = N_L

    Trans = l2_calculate_transmission(H_full, S_full, Erange, N_L, N_R)

    np.savetxt('Transmission.txt', np.column_stack((Erange, np.abs(Trans))),
               fmt='%10.5f\t%12.8e', delimiter='\t', header='# Erange\t# Transmission', comments='')

    matplotlib.use('Agg')
    plt.figure()
    plt.plot(Erange, np.log10(np.abs(Trans)), label='Transmission', color='blue')
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
    calculate_transmission(xyz_filename, chosen_method)

if __name__ == "__main__":
    main()
