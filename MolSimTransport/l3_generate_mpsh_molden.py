import numpy as np
import json
from importlib.resources import files
import MolSimTransport.utils.get_gfn1_GTOinfo_from_json as gto_info

# Preload atomic basis function counts and new order data
gfn1_basis_functions = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('gfn1_ao_num.json'), 'r'))
new_order = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('gfn1_xtb2molden_ao_reorder.json'), 'r'))

def generate_mo_section(eigenenergies_file, eigenvectors_file, atom_order_file):
    """
    Generates the [MO] section for the Molden file, including orbital energies, occupancies, and coefficients.
    """
    energies = np.loadtxt(eigenenergies_file, skiprows=3)
    coeff_vectors = np.loadtxt(eigenvectors_file, skiprows=1)
    occupations = [2.0] * len(energies)  # for reference only, not related to the actual occupancy
    atom_order = read_atom_order_file(atom_order_file)
    total_basis = sum([gfn1_basis_functions[atom] for atom in atom_order])

    if coeff_vectors.shape[0] != total_basis:
        raise ValueError("The number of rows in the coefficient vector does not match the total basis functions. Check input files.")

    ev_to_au = 0.0367493
    mo_section = "[MO]\n"

    for i in range(coeff_vectors.shape[1]):
        single_coeff_vector = coeff_vectors[:, i]
        new_coeff_vector = rearrange_orbital_coefficients(atom_order, single_coeff_vector)

        mo_section += f" Sym= {i+1}A\n"
        mo_section += f" Ene= {energies[i] * ev_to_au:.6f}\n"
        mo_section += f" Spin= Alpha\n"
        mo_section += f" Occup= {occupations[i]:.6f}\n"

        for j, coeff in enumerate(new_coeff_vector):
            mo_section += f" {j+1:2d} {coeff:22.16f}\n"
    
    return mo_section

def rearrange_orbital_coefficients(atom_order, coeff_vector):
    """
    Rearranges the orbital coefficient vector based on atom order and the new basis function order.
    """
    new_coeff_vector = np.zeros(len(coeff_vector))
    current_idx = 0
    new_idx = 0

    for atom in atom_order:
        num_basis = gfn1_basis_functions.get(atom, 0)
        original_indices = np.arange(current_idx, current_idx + num_basis)
        
        if atom in new_order:
            new_indices = new_order[atom]
            if len(new_indices) != num_basis:
                raise ValueError(f'The new order for atom {atom} does not match the number of basis functions.')
        else:
            new_indices = np.arange(num_basis)

        for j in range(num_basis):
            new_coeff_vector[new_idx] = coeff_vector[original_indices[new_indices[j] - 1]]
            new_idx += 1
        current_idx += num_basis
    
    return normalize_vector(new_coeff_vector)

def read_atom_order_file(filename):
    """
    Reads the atom order from a file.
    """
    data = np.loadtxt(filename, dtype=str, skiprows=1, usecols=0)
    return data.tolist()

def normalize_vector(vector):
    """
    Normalizes a vector.
    """
    norm_value = np.linalg.norm(vector)
    return vector / norm_value if norm_value != 0 else vector

def generate_atoms_section(em_file):
    """
    Generates the [Atoms] section.
    """
    atoms_section = "[Atoms] Angs\n"
    with open(em_file, 'r') as f:
        lines = f.readlines()[1:]
        for atom_index, line in enumerate(lines, start=1):
            parts = line.split()
            element, atomic_num, x, y, z = parts[0], parts[2], parts[3], parts[4], parts[5]
            atoms_section += f"{element:<4}{atom_index:<6}{atomic_num:<6}{x:<12}{y:<12}{z:<12}\n"
    return atoms_section

def generate_gto_section(json_file, atoms_file):
    """
    Generates the [GTO] section.
    """
    gto_section = "[GTO]\n"
    with open(atoms_file, 'r') as f:
        for atomic_index, line in enumerate(f.readlines()[1:], start=1):
            atomic_num = line.split()[2]
            gto_section += f"          {atomic_index} 0\n"
            basis_set_info = gto_info.get_basis_set_by_atomic_number(json_file, atomic_num)
            if basis_set_info:
                gto_section += f"{basis_set_info}\n\n"
    return gto_section

def generate_molden_file(em_file, eigenenergies_file, eigenvectors_file, json_file, output_file):
    """
    Generates the complete Molden file.
    """
    title_section = "[Molden Format]\n[Title]\n[5D]\n"
    atoms_section = generate_atoms_section(em_file)
    gto_section = generate_gto_section(json_file, em_file)
    mo_section = generate_mo_section(eigenenergies_file, eigenvectors_file, em_file)

    with open(output_file, 'w') as f:
        f.write(title_section)
        f.write(atoms_section)
        f.write(gto_section)
        f.write(mo_section)


def main():
    eigenenergies_file = 'mpsh_eigenvalues.txt'
    eigenvectors_file = 'mpsh_eigenvectors.txt'
    em_file = 'EM_atoms.txt'
    atomic_orbitals_json_file = files('MolSimTransport.utils.atomic_data').joinpath('gfn1_gto_basis.json')
    output_file = 'MPSH.molden'
    generate_molden_file(em_file, eigenenergies_file, eigenvectors_file, atomic_orbitals_json_file, output_file)

    print("MPSH.molden file has been generated!")

if __name__ == "__main__":
    main()