from tblite.interface import Calculator
import numpy as np
import sys
import pandas as pd

def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    natoms = int(lines[0].strip())
    
    numbers = []
    positions = []
    
    for line in lines[2:2+natoms]:
        parts = line.strip().split()
        atom = parts[0]
        coords = list(map(float, parts[1:4]))
        
        atom_to_number = {
            'H': 1,
            'He': 2,
            'Li': 3,
            'Be': 4,
            'B': 5,
            'C': 6,
            'N': 7,
            'O': 8,
            'S': 16,
            'Au': 79
        }
        
        numbers.append(atom_to_number[atom])
        positions.append(coords)
    
    return np.array(numbers), np.array(positions)

numbers, positions_ang = read_xyz('pyridine-pyramid-angle90.xyz')
print(positions_ang)
positions = 1.8897161646320724 * positions_ang
print(positions)

calc = Calculator(
    method="GFN1-xTB",
    numbers=numbers,
    positions=positions
)

with open('basis-pyridine-pyramid-angle90-gfn1.txt', 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f

    print(calc.get("shell-map"))
    print(calc.get("angular-momenta"))
    print(calc.get("orbital-map"))
    
    sys.stdout = original_stdout

calc.set("save-integrals", 1)
res = calc.singlepoint()


df = pd.DataFrame(res.get("hamiltonian-matrix"))
df.to_csv('pyridine-pyramid-angle90-gfn1_hamiltonian_matrix_xtb.csv', index=False)

do = pd.DataFrame(res.get("overlap-matrix"))
do.to_csv('pyridine-pyramid-angle90-gfn1_overlap_matrix_xtb.csv', index=False)

de =27.2114 * pd.DataFrame(res.get("orbital-energies"))
de.to_csv('pyridine-pyramid-angle90-gfn1_orbital-energies_xtb.csv', index=False)

dp = pd.DataFrame(res.get("orbital-occupations"))
dp.to_csv('pyridine-pyramid-angle90-gfn1_orbital-occupations_xtb.csv', index=False)

dc = pd.DataFrame(res.get("orbital-coefficients"))
dc.to_csv('pyridine-pyramid-angle90-gfn1_orbital-coefficients_xtb.csv', index=False)
