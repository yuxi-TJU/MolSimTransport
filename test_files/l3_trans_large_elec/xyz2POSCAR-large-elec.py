import numpy as np
import json
from importlib.resources import files

atom_to_number = json.load(open(files('MolSimTransport.utils.atomic_data').joinpath('atomic_num.json'), 'r'))

def read_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = []
    for line in lines[2:num_atoms + 2]:
        atom_data = line.split()
        atoms.append({
            'element': atom_data[0],
            'x': float(atom_data[1]),
            'y': float(atom_data[2]),
            'z': float(atom_data[3])
        })
    z_coords = [atom['z'] for atom in atoms]
    z_length = max(z_coords) - min(z_coords)
    return atoms, comment, z_length

def write_poscar(atoms, comment, lattice_vectors, filename='POSCAR'):
    total_written_atoms = 0  
    poscar_atoms = [] # Records the order of atoms in the POSCAR file
    with open(filename, 'w') as f:
        f.write(comment + '\n')
        f.write('1.0\n')
        for vec in lattice_vectors:
            f.write('  '.join(f"{v:16.12f}" for v in vec) + '\n')
        elements = [atom['element'] for atom in atoms]
        new_elements = []
        element_counts = []
        for element in elements:
            if element not in new_elements:
                new_elements.append(element)
                element_counts.append(1)
            else:
                element_counts[new_elements.index(element)] += 1

        # Split Au atoms, move half of them to the end
        if 'Au' in new_elements:
            au_index = new_elements.index('Au')
            total_au_count = element_counts[au_index]
            half_au_count = total_au_count // 2
            new_elements.append('Au')
            element_counts.append(half_au_count)
            element_counts[au_index] = total_au_count - half_au_count

        f.write('  '.join(new_elements) + '\n')
        f.write('  '.join(map(str, element_counts)) + '\n')
        f.write('Direct\n')
        au_atoms = [atom for atom in atoms if atom['element'] == 'Au']
        for atom in au_atoms[:len(au_atoms)//2]:  
            f.write(f"{atom['fx']:16.12f} {atom['fy']:16.12f} {atom['fz']:16.12f}\n")
            poscar_atoms.append(atom)  # Recording atomic order
            total_written_atoms += 1
        
        for element in new_elements:
            if element != 'Au':  
                for atom in atoms:
                    if atom['element'] == element:
                        f.write(f"{atom['fx']:16.12f} {atom['fy']:16.12f} {atom['fz']:16.12f}\n")
                        poscar_atoms.append(atom)  
                        total_written_atoms += 1
        for atom in au_atoms[len(au_atoms)//2:]:  
            f.write(f"{atom['fx']:16.12f} {atom['fy']:16.12f} {atom['fz']:16.12f}\n")
            poscar_atoms.append(atom) 
            total_written_atoms += 1
    return total_written_atoms, poscar_atoms  # Returns the number of atoms written and the order of atoms in POSCAR

def cartesian_to_fractional(atoms, lattice_vectors):
    lattice_matrix = np.array(lattice_vectors)
    inverse_lattice = np.linalg.inv(lattice_matrix)
    fractional_coords = []
    for atom in atoms:
        cartesian_coords = np.array([atom['x'], atom['y'], atom['z']])
        fractional_coords.append(np.dot(inverse_lattice, cartesian_coords))
    fractional_coords = np.array(fractional_coords)
    z_center = np.mean(fractional_coords[:, 2])
    z_shift = 0.5 - z_center 
    # Translate the fractional coordinates in the z direction for all atoms and filter out atoms that are not in the range [0, 1)
    filtered_atoms = []
    for i, atom in enumerate(atoms):
        fractional_coords[i][2] += z_shift
        if all(0 <= coord < 1 for coord in fractional_coords[i]):
            atom['fx'], atom['fy'], atom['fz'] = fractional_coords[i]
            filtered_atoms.append(atom)
    return filtered_atoms

# Write the extracted middle atoms to a file with additional information
def extract_middle_part(poscar_atoms, filename='EM_atoms.txt'):
    middle_atoms = poscar_atoms[108:-108] 
    with open(filename, 'w') as f:
        f.write(f"{'Element':<8}{'Seq#':<8}{'AtomicNum':<12}{'x (Ang)':<14}{'y (Ang)':<14}{'z (Ang)'}\n")
        for i, atom in enumerate(middle_atoms):
            atomic_number = atom_to_number[atom['element']]
            f.write(f"{atom['element']:<8}{i + 1:<8}{atomic_number:<8}"
                    f"{atom['x']:14.8f}{atom['y']:14.8f}{atom['z']:14.8f}\n")


#############################################################################################
##############  Enter the name of the XYZ file to be converted to POSCAR here  ##############
xyz_filename = 'junction_example_pyramid_Rede_1.xyz'   # Change to your XYZ file         
atoms, comment, z_length = read_xyz(xyz_filename)

################  Define the vacuum length of the system in the z-direction  ################
lattice_z = z_length + 20   # Change the vacuum length in the z-direction
lattice_vectors = [  
    [25.953810, 0.0, 0.0],  # Example lattice vector 
    [0.0, 9.9896300, 0.0],  # periodic in xy direction 
    [0.0, 0.0, lattice_z]] 
#############################################################################################


# cartesian_to_fractional and move_to_center_in_z_direction
new_atoms = cartesian_to_fractional(atoms, lattice_vectors)

# Writes to the POSCAR file and returns the number of atoms written and the order of atoms in POSCAR
output_atom_count, poscar_atoms = write_poscar(new_atoms, comment, lattice_vectors, 'POSCAR')

# Extract the middle part of the atoms (excluding top and bottom 36 atoms)
extract_middle_part(poscar_atoms, 'EM_atoms.txt')

# Check if the number of atoms in the generated POSCAR file is 20 less than the input xyz file
input_atom_count = len(atoms)

print("xyz atom number: ", input_atom_count)
print("POSCAR atom number:", output_atom_count)

if input_atom_count - output_atom_count == 0:
    print("Conversion completed! The structure is periodic in the xy direction.")
    print("Please double-check the generated POSCAR file for correctness.")
else:
    print("Something wrong! The number of atoms in the generated poscar file did not meet expectations!")
