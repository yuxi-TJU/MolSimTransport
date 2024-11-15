import numpy as np
import json
from importlib.resources import files
import MolSimTransport.utils.get_gfn1_GTOinfo_from_json as gto_info

# Preload atomic basis function counts and new order data
gfn1_basis_functions = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('gfn1_ao_num.json'), 'r'))
new_order = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('gfn1_xtb2molden_ao_reorder.json'), 'r'))


def generate_EC_molden(closest_energy, trans_eigenvector):
    eigenvectors = trans_eigenvector
    em_file = 'EM_atoms.txt'
    atomic_orbitals_json_file = files('MolSimTransport.utils.atomic_data').joinpath('gfn1_gto_basis.json')
    generate_molden_file(em_file, closest_energy, eigenvectors, atomic_orbitals_json_file)

def generate_mo_section(trans_eigenvector, atom_order_file):
    """
    Generates the [MO] section for the Molden file, including orbital energies, occupancies, and coefficients.
    """

    coeff_vector = trans_eigenvector
    occupations = 2.0  # for reference only, not related to the actual occupancy
    atom_order = read_atom_order_file(atom_order_file)
    total_basis = sum([gfn1_basis_functions[atom] for atom in atom_order])

    if coeff_vector.shape[0] != total_basis:
        raise ValueError("The number of rows in the coefficient vector does not match the total basis functions. Check input files.")

    absolute_coeff = np.abs(coeff_vector)
    real_coeff = np.real(coeff_vector)
    imag_coeff = np.imag(coeff_vector)

    reorder_absolute_coeff = rearrange_orbital_coefficients(atom_order, absolute_coeff)
    reorder_real_coeff = rearrange_orbital_coefficients(atom_order, real_coeff)
    reorder_imag_coeff = rearrange_orbital_coefficients(atom_order, imag_coeff)

    return occupations, reorder_absolute_coeff, reorder_real_coeff, reorder_imag_coeff

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

def generate_molden_file(em_file, closest_energy, eigenvectors, json_file):
    """
    Generates the complete Molden file.
    """
    title_section = "[Molden Format]\n[Title]\n[5D]\n"
    atoms_section = generate_atoms_section(em_file)
    gto_section = generate_gto_section(json_file, em_file)
    ev_to_au = 0.0367493
    closest_energy_str = f"{closest_energy:.5f}"

    occupation, absolute_coeff, real_coeff, imag_coeff = generate_mo_section(eigenvectors, em_file)

    output_files = {
        "Absolute": f"EC_Abs_{closest_energy_str}.molden",
        "Real": f"EC_Re_{closest_energy_str}.molden",
        "Imag": f"EC_Im_{closest_energy_str}.molden"
    }

    for coeffs, coeff_type in zip([absolute_coeff, real_coeff, imag_coeff], output_files):
        mo_section = "[MO]\n"
        mo_section += f" Sym= 1A\n"
        mo_section += f" Ene= {closest_energy * ev_to_au:.6f}\n"
        mo_section += f" Spin= Alpha\n"
        mo_section += f" Occup= {occupation}\n"

        for j, coeff in enumerate(coeffs):
            mo_section += f" {j+1:2d} {coeff:22.16f}\n"
        
        with open(output_files[coeff_type], 'w') as f:
            f.write(title_section)
            f.write(atoms_section)
            f.write(gto_section)
            f.write(mo_section)
        
    print("EigenChannel's Molden files have been generated!")
