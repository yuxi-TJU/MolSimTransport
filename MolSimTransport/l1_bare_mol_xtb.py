'''
A python script that computes the transport of isolated molecules under the xTB method.
'''

from tblite.interface import Calculator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import argparse
import time

def run_xtb_wba():
    options = get_options(sys.argv[1:])
    main(options)

def get_options(argv):
    parser = argparse.ArgumentParser(description='L1_XTB Module: xTB method + WBA for isolated molecule')
    parser.add_argument('-f', '--file', type=str, help='Input xyz-file for calculations')
    parser.add_argument('-L', '--left', type=int, nargs='+', help='Input left atom numbers (separated by space)')
    parser.add_argument('-R', '--right', type=int, nargs='+', help='Input right atom numbers (separated by space)')
    parser.add_argument('-C', '--coupling', type=float, default=1.0, help='Input GammaL_wba (The default is GammaL_wba = GammaR_wba) [default: %(default)s]')
    parser.add_argument('-m', '--method', type=int, choices=[1, 2], default=1, help='Specify the calculation method: 1 for GFN1-xTB, 2 for GFN2-xTB [default: %(default)s]')
    parser.add_argument('--Erange', type=float, default=4.0, help='Specify the energy range (from \'Fermi energy\' plus or minus \'Erange\')[default: %(default)s eV]')
    parser.add_argument('--Enum', type=int, default=800, help='Specify the energy number [default: %(default)s]')
    parser.add_argument('--interactive', action='store_true', help='Use interactive mode to input parameters')
    return parser.parse_args(argv)

def read_xyz(file):
    atom_to_number = { 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
                        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
                        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
                        'Cs': 55, 'Ba': 56, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86 }
    with open(file, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    numbers, positions, elements = [], [], []
    for line in lines[2:2+natoms]:
        parts = line.split()
        atom = parts[0]
        coords = list(map(lambda x: float(x) / 0.52917721067, parts[1:4]))  # Ang to Bohr
        numbers.append(atom_to_number[atom])
        elements.append(atom)
        positions.append(coords)
    return np.array(numbers), np.array(positions), elements

def calculate_orbital_indexes(atom_numbers, elements, atomic_orbitals):
    indexes, total_orbitals = [], 0
    for i, atom in enumerate(elements):
        if i+1 in atom_numbers:
            indexes.append((total_orbitals, total_orbitals + atomic_orbitals[atom]))
        total_orbitals += atomic_orbitals[atom]
    return indexes, total_orbitals

def build_coupling_matrix(indexes, wba, N, dtype=float):
    matrix = np.zeros((N, N), dtype=dtype)
    for start, end in indexes:
        matrix[start:end, start:end] = np.diag([wba] * (end - start))
    return matrix

def main(options):
    start_time = time.time()
    
    if options.interactive:
        options.file = input(">>> Enter XYZ file name (e.g., coor.xyz): ")
        options.left = [int(n) for n in input(">>> Enter left atom numbers (separated by space): ").split()]
        options.right = [int(n) for n in input(">>> Enter right atom numbers (separated by space): ").split()]
        options.coupling = float(input(">>> Specify the coupling strength (e.g., 1): "))
        options.method = int(input(">>> Enter calculation method (1 for GFN1-xTB, 2 for GFN2-xTB, e.g., 1): "))
        options.Erange = float(input(">>> (from 'Fermi energy' plus or minus 'energy range', e.g., 4): "))
        options.Enum = int(input(">>> Specify the energy number (e.g., 800): "))

    numbers, positions, elements = read_xyz(options.file)
    method_mapping = {1: "GFN1-xTB", 2: "GFN2-xTB"}
    chosen_method = method_mapping[options.method]
    
    calc = Calculator(method=chosen_method, numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    res = calc.singlepoint()
    
    OrbE = 27.2114 * res.get("orbital-energies")
    S = res.get("overlap-matrix")
    C = np.transpose(res.get("orbital-coefficients"))
    E = np.diag(OrbE.flatten())
    OCC = res.get("orbital-occupations")
    H_cec = C @ E @ np.linalg.inv(C)  # H_cec = inv(S)*H
    H_full = S @ H_cec

    LUMO_index = np.where(OCC < 0.001)[0][0]
    FermiEnergy = (OrbE[LUMO_index] + OrbE[LUMO_index-1]) / 2
    E_num = options.Enum
    Erange = np.linspace(FermiEnergy - options.Erange, FermiEnergy + options.Erange, E_num)

    atomic_orbitals1 = {'H': 2, 'He': 1, 'Li': 4, 'Na': 4, 'K': 4, 'Rb': 4, 'Cs': 4, 
                        'Be': 4, 'B': 4, 'C': 4, 'N': 4, 'O': 4, 'F': 4, 'Zn': 4, 'Cd': 4, 'Hg': 4, 
                        'Tl': 4, 'Pb': 4, 'Bi': 4, 'Po': 9, 'Al': 9, 'Si': 9, 'P': 9, 'S': 9, 'Ag': 9, 'Au': 9}
    atomic_orbitals2 = {'H': 1, 'He': 4, 'Li': 4, 'Na': 4, 'K': 4, 'Rb': 4, 'Cs': 4, 
                        'Be': 4, 'B': 4, 'C': 4, 'N': 4, 'O': 4, 'F': 4, 'Zn': 4, 'Cd': 4, 'Hg': 4, 
                        'Tl': 4, 'Pb': 4, 'Bi': 4, 'Po': 4, 'Al': 9, 'Si': 9, 'P': 9, 'S': 9, 'Ag': 9, 'Au': 9}
    
    atomic_orbitals = atomic_orbitals1 if chosen_method == "GFN1-xTB" else atomic_orbitals2
    left_indexes, _ = calculate_orbital_indexes(options.left, elements, atomic_orbitals)
    right_indexes, N = calculate_orbital_indexes(options.right, elements, atomic_orbitals)

    GammaL_wba = GammaR_wba = options.coupling
    SigmaL_wba = SigmaR_wba = -0.5j * GammaL_wba
    z_plus = 1e-13j
    Trans = np.zeros(E_num, dtype=complex)

    print("Starting transport calculation...")

    for i in range(E_num):
        GammaL = build_coupling_matrix(left_indexes, GammaL_wba, N)
        GammaR = build_coupling_matrix(right_indexes, GammaR_wba, N)
        SigmaL = build_coupling_matrix(left_indexes, SigmaL_wba, N, dtype=complex)
        SigmaR = build_coupling_matrix(right_indexes, SigmaR_wba, N, dtype=complex)

        G = np.linalg.inv((Erange[i] + z_plus) * S - H_full - SigmaL - SigmaR)
        Trans_list = G @ GammaL @ G.T.conj() @ GammaR
        Trans[i] = np.real(np.trace(Trans_list))
    
    np.savetxt('Transmission.txt', np.column_stack((Erange, np.abs(Trans))), fmt='%12.8f\t%12.8f', delimiter='\t', header='Erange\tTransmission', comments='')

    matplotlib.use('Agg')
    plt.figure()
    y_fixed = 1.0 
    OrbE_rmfo = [value for value in OrbE if value not in (OrbE[LUMO_index-1], OrbE[LUMO_index])]
    plt.scatter(OrbE_rmfo, [y_fixed] * len(OrbE_rmfo), marker='o', edgecolors='b', linewidth=1, s=50)
    plt.scatter((OrbE[LUMO_index-1], OrbE[LUMO_index]), [y_fixed] * 2, marker='o', edgecolors='r', linewidth=1, s=50)
    plt.plot(Erange, np.log10(np.abs(Trans)), label='Transmission', color='blue')
    plt.title('Transmission Spectrum (WBA)')
    plt.xlim([FermiEnergy - options.Erange, FermiEnergy + options.Erange])
    plt.xlabel('Energy (eV)', fontweight='bold')
    plt.ylabel('Transmission (log10)', fontweight='bold')
    plt.legend()
    plt.savefig('Transmission.png')

    print(" ")
    print(f"Chosen method:  {chosen_method}")
    print("Total orbitals: ", N)
    print("Indexes of the atomic basis on the left (start, end): ", [(x + 1, y) for (x, y) in left_indexes])
    print("Indexes of the atomic basis on the right (start, end): ", [(x + 1, y) for (x, y) in right_indexes])
    print("\"Fermi energy\" (Defined as the average of the HOMO and LUMO energies) (eV): ",FermiEnergy)
    print(" ")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print("Done!")

if __name__ == "__main__":
    options = get_options(sys.argv[1:])
    main(options)
