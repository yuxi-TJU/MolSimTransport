'''
A python script that computes the transport of isolated molecules under the EHT method.
'''

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdEHTTools
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import argparse
import sys
import matplotlib
import json
from importlib.resources import files
import time
from MolSimTransport.utils.functions import l1_calculate_transmission

def run_eht_wba():
    options = get_options(sys.argv[1:])
    main(options)

def get_options(argv):
    parser = argparse.ArgumentParser(description='L1_EHT Module: EHT method + WBA for isolated molecule')
    parser.add_argument('-f', '--file', type=str, help='Input xyz-file for calculations')
    parser.add_argument('-L', '--left', type=int, nargs='+', help='Input left atom numbers (separated by space)')
    parser.add_argument('-R', '--right', type=int, nargs='+', help='Input right atom numbers (separated by space)')
    parser.add_argument('-C', '--coupling', type=float, default=0.1, help='Input GammaL_wba [default: %(default)s]')
    parser.add_argument('--Erange', type=float, nargs=2, default=[-15, -6], help='Specify the energy range [default: %(default)s eV]')
    parser.add_argument('--Enum', type=int, default=900, help='Specify the energy number [default: %(default)s]')
    parser.add_argument('--interactive', action='store_true', help='Use interactive mode to input parameters')
    return parser.parse_args(argv)

def read_xyz(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())
    mol = Chem.RWMol()
    conf = Chem.Conformer(num_atoms)
    elements = []
    for i, line in enumerate(lines[2:2 + num_atoms]):
        parts = line.strip().split()
        atom = Chem.Atom(parts[0])
        mol.AddAtom(atom)
        elements.append(parts[0])
        conf.SetAtomPosition(i, tuple(map(float, parts[1:4])))
    mol.AddConformer(conf)
    return mol, elements

def symmetrize(matrix):
    return np.triu(matrix) + np.triu(matrix, k=1).T

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
        options.coupling = float(input(">>> Specify the coupling strength (e.g., 0.1): "))
        options.Erange = [float(n) for n in input(">>> Specify the energy range (two values separated by space, e.g., -15 -6): ").split()]
        options.Enum = int(input(">>> Specify the energy number (e.g., 900): "))

    mol, elements = read_xyz(options.file)
    res = rdEHTTools.RunMol(mol, keepOverlapAndHamiltonianMatrices=True)
    H = symmetrize(res[1].GetHamiltonian())
    S = symmetrize(res[1].GetOverlapMatrix())
    num_electrons = res[1].numElectrons

    Erange = np.linspace(*options.Erange, options.Enum)
    eigenvalues, eigenvectors = eigh(H, S)
    eigenvalues.sort()
    homo_index = num_electrons // 2 + (num_electrons % 2)
    print(" ")
    print(f"HOMO index: {homo_index}")
    print(f"HOMO energy (eV): {eigenvalues[homo_index - 1]}")
    print(f"LUMO index: {homo_index + 1}")
    print(f"LUMO energy (eV): {eigenvalues[homo_index]}")

    atomic_orbitals = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('eht_ao_num.json'), 'r'))

    left_indexes, _ = calculate_orbital_indexes(options.left, elements, atomic_orbitals)
    right_indexes, N = calculate_orbital_indexes(options.right, elements, atomic_orbitals)
    GammaL_wba = GammaR_wba = options.coupling
    Trans = np.zeros(options.Enum, dtype=complex)

    Trans = l1_calculate_transmission(Erange, S, H, left_indexes, right_indexes, GammaL_wba, GammaR_wba)

    np.savetxt('Transmission.txt', np.column_stack((Erange, np.abs(Trans))), fmt='%10.5f\t%12.8e', delimiter='\t', header='Erange\tTransmission', comments='')

    matplotlib.use('Agg')
    plt.figure()
    y_fixed = 1.0
    eigenvalues_rmfo = [value for value in eigenvalues if value not in (eigenvalues[homo_index-1], eigenvalues[homo_index])]
    plt.scatter(eigenvalues_rmfo, [y_fixed] * len(eigenvalues_rmfo), marker='o', edgecolors='b', linewidth=1, s=50)
    plt.scatter((eigenvalues[homo_index-1], eigenvalues[homo_index]), [y_fixed] * 2, marker='o', edgecolors='r', linewidth=1, s=50)
    plt.xlim(*options.Erange)
    np.seterr(divide='ignore')
    plt.plot(Erange, np.log10(np.abs(Trans)), label='Transmission', color='blue')
    plt.xlabel('Energy (eV)', fontweight='bold')
    plt.ylabel('Transmission (log10)', fontweight='bold')
    plt.title('Transmission Spectrum (WBA)')
    plt.savefig('Transmission.png')
    print(" ")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print("Done!")

if __name__ == "__main__":
    options = get_options(sys.argv[1:])
    main(options)
