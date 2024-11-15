import os
import numpy as np
from scipy.constants import e, h, k
from scipy.integrate import quad
import subprocess
import time

def create_and_run_for_field(base_dir, poscar_file, eefs, input_energy_range, input_energy_interval, save_mat_files):
    eefd = '1' if eefs > 0 else '-1'
    eefs = f"{eefs:.4f}"
    
    dir_name = os.path.join(base_dir, f"Field_{eefs}")  # Use absolute path
    os.makedirs(dir_name, exist_ok=True) 
    
    # Copy files to the target directory
    poscar_dest = os.path.join(dir_name, poscar_file)
    subprocess.run(["cp", os.path.join(base_dir, poscar_file), poscar_dest], check=True)
    
    print(" ")
    print(f"Calculating for electric field: {eefs}")
    print(f"Creating directory '{dir_name}' and run DFTB+ calculation...")

    # Run the calculation script in the target directory
    subprocess.run([
        "L3_EEF", poscar_dest, str(eefs), eefd, str(input_energy_range), str(input_energy_interval), dir_name, str(save_mat_files) 
    ], cwd=dir_name, check=True)

def merge_transmission_files(base_dir, electric_field_range):
    combined_data = None
    
    for eefs in electric_field_range:
        eefs = f"{eefs:.4f}"
        dir_name = os.path.join(base_dir, f"Field_{eefs}")
        file_path = os.path.join(dir_name, "Transmission.txt")
        
        if os.path.exists(file_path):
            data = np.loadtxt(file_path, skiprows=1)
            
            if combined_data is None:
                combined_data = data
            else:
                if combined_data.shape[0] != data.shape[0]:
                    raise ValueError(f"File {file_path} has inconsistent data length.")
                combined_data = np.column_stack((combined_data, data))
    
    np.savetxt(os.path.join(base_dir, "combined_transmission.txt"), combined_data, fmt='%22.15f', delimiter='\t')

def main():
    start_time = time.time()
    base_dir = os.getcwd()  # Get the current working directory (base directory)

########################################################################################################################
    poscar_file = "10fc3.POSCAR"  # Modify to your POSCAR file name
    Length = 26.67057  # Modify to your extended molecule length (in Angstrom)
    save_mat_files = False  # Don't generate Gr, GammaL, GammaR mat files. Change to True if needed
    input_energy_range = 2  # Energy range in eV (float)
    input_energy_interval = 0.01  # Energy interval (float)
    electric_field_range = np.arange(-0.0004, 0.0005, 0.0001)  # Electric field range in atomic units(a.u.)
    # electric_field_range = np.array([0.0006])  # If only a single bias point is calculated, comment out the line above

    # Remove zero electric field value if present
    electric_field_range = electric_field_range[~np.isclose(electric_field_range, 0, atol=1e-10)]

    # Sequentially execute each task for each electric field value
    for eefs in electric_field_range:
        create_and_run_for_field(base_dir, poscar_file, eefs, input_energy_range, input_energy_interval, save_mat_files)
########################################################################################################################

    # Merge the transmission files
    merge_transmission_files(base_dir, electric_field_range)

    # Calculate current
    T = 300
    kT = k * T / e

    data = np.loadtxt(os.path.join(base_dir, 'combined_transmission.txt'))
    V = np.round(electric_field_range * Length * 51.42, 3)
    currents = np.zeros_like(V)

    def fermi(E, mu):
        return 1 / (np.exp((E - mu) / kT) + 1)

    for i, bias in enumerate(V):
        E = data[:, 2*i]
        T = data[:, 2*i+1]
        
        E = np.asarray(E).flatten()
        T = np.asarray(T).flatten()

        T_interp = lambda E_query: np.interp(E_query, E, T)
    
        mu1 = bias / 2
        mu2 = -bias / 2

        def integrand(E):
            return T_interp(E) * (fermi(E, mu1) - fermi(E, mu2))

        currents[i], _ = quad(integrand, min(E), max(E))
        currents[i] *= 2 * e**2 / h

    output_data = np.column_stack((V, currents))  
    np.savetxt(os.path.join(base_dir, 'voltage_current.txt'), output_data, fmt='%8.4f\t%15.8e', header='Voltage(V) Current(A)', comments='')

    elapsed_time = (time.time() - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print("All electric field calculations are completed!")
    print("Generation of current-voltage data files is completed!")

if __name__ == "__main__":
    main()
