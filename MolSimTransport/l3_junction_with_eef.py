'''
A python script computes the transport of molecular junction containing the PL under the electric field.
'''

import os
import numpy as np
import math
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib
import sys
from importlib.resources import files
from MolSimTransport.utils.functions import generate_1st_dftb_eef, generate_2nd_dftb_eef, interpolate_sgf_matrices, l3_calculate_transmission_eef
import time

start_time = time.time()

def main():
    # Fetch command-line arguments
    poscar_file = sys.argv[1]  # First argument: POSCAR file
    eefs = float(sys.argv[2])  # Second argument: Electric field strength
    eefd = sys.argv[3]         # Third argument: Electric field direction
    InputEnergyRange = float(sys.argv[4])  # Fourth argument: Energy range
    InputEnergyInterval = float(sys.argv[5])  # Fifth argument: Energy interval
    output_dir = sys.argv[6]  # Sixth argument: Output file path
    save_mat_files = sys.argv[7].lower() == 'true'  # Seventh argument: Whether to generate mat files

    # Define the path to the POSCAR file, uniform electric field strength and direction
    # poscar_file = str(input("Enter POSCAR file name(e.g., coor.POSCAR): "))
    # eefs = float(input("Enter Uniform electric field strength: "))
    # eefd = input("Enter Uniform electric field direction: ")
    dftb_input_file = "dftb_in.hsd"
    dftb_output_file = "dftb_output.out"

    # Generate the DFTB+ input file for step 1
    generate_1st_dftb_eef(poscar_file, eefs, eefd, dftb_input_file)

    # Run the DFTB+ program
    os.system("dftb+ > " + dftb_output_file)
    os.rename("dftb_in.hsd", "1st_dftb_in.hsd")

    # Generate the DFTB+ input file for step 2
    generate_2nd_dftb_eef(poscar_file, dftb_input_file)

    # Run the DFTB+ program
    os.system("dftb+ > " + dftb_output_file)

    # Rename the hamsqr1.dat and oversqr.dat files
    if os.path.exists("hamsqr1.dat"):
        os.rename("hamsqr1.dat", f"device_h_{eefd}_{eefs:.4f}.dat")
    else:
        print("file not found: hamsqr1.dat")

    if os.path.exists("oversqr.dat"):
        os.rename("oversqr.dat", f"device_s_{eefd}_{eefs:.4f}.dat")
    else:
        print("file not found: oversqr.dat")
    
    os.rename("dftb_in.hsd", "2nd_dftb_in.hsd")
    print(" ")
    print("DFTB+ calculation done!")

    # Load electrode data
    h1_file_path = files('share.l3_elec').joinpath('h1-kpoint-avg.dat')
    h1_2pl_data = np.loadtxt(h1_file_path)
    h1_2pl_kavg = h1_2pl_data[:, 0::2] + 1j * h1_2pl_data[:, 1::2]

    h2_file_path = files('share.l3_elec').joinpath('h2-kpoint-avg.dat')
    h2_2pl_data = np.loadtxt(h2_file_path)
    h2_2pl_kavg = h2_2pl_data[:, 0::2] + 1j * h2_2pl_data[:, 1::2]

    efermi = -12.3377

    # Find and load the 'device_h_*' file and 'device_s_*' file.
    h_file_name = next((f for f in os.listdir() if f.startswith('device_h_') and f.endswith('.dat')), None)
    if not h_file_name:
        raise FileNotFoundError("No file found starting with 'device_h_' and ending with '.dat'")
    h = 27.2114 * np.loadtxt(h_file_name, skiprows=5)

    s_file_name = next((f for f in os.listdir() if f.startswith('device_s_') and f.endswith('.dat')), None)
    if not s_file_name:
        raise FileNotFoundError("No file found starting with 'device_s_' and ending with '.dat'")
    s = np.loadtxt(s_file_name, skiprows=5)

    npl = len(h1_2pl_kavg) // 2
    nc = len(h)
    nm = nc - 2 * npl

    # Partition matrix
    h_l = h[:npl, :npl]
    h_ml = h[npl:npl+nm, :npl]
    h_mr = h[npl:npl+nm, nc-npl:]
    h_r = h[nc-npl:, nc-npl:]
    h_m = h[npl:npl+nm, npl:npl+nm]

    s_l = s[:npl, :npl]
    s_ml = s[npl:npl+nm, :npl]
    s_mr = s[npl:npl+nm, nc-npl:]
    s_r = s[nc-npl:, nc-npl:]
    s_m = s[npl:npl+nm, npl:npl+nm]

    # Calculate the Fermi level offset
    sum_l = 0
    sum_r = 0

    for j in range(npl):
        sum_l += (h1_2pl_kavg[j, j] - h_l[j, j]) / s_l[j, j]
        sum_r += (h2_2pl_kavg[j, j] - h_r[j, j]) / s_r[j, j]

    phi = (sum_l + sum_r) / (2 * npl)

    offset_efermi = round(np.real(phi - efermi), 5)

    mpsh_w, mpsh_v = eigh(h_m, s_m)
    mpsh_w += offset_efermi

    with open("mpsh_eigenvalues.txt", "w") as f:
        f.write("# The offset of Fermi energy\n")
        f.write(f"{offset_efermi}\n")
        f.write("# mpsh_eigenvalues\n")
        np.savetxt(f, mpsh_w, fmt='%10.5f')

    with open("mpsh_eigenvectors.txt", "w") as f:
        f.write("# mpsh_eigenvectors\n")
        np.savetxt(f, mpsh_v, fmt='%22.16f') 

    # Select specific energy points and interpolations
    specific_energy_points = np.arange(-15, -4, 1)

    mat_file_path = files('share.l3_elec').joinpath('sgf_k10_specific_results_15to5_0.1.mat')
    pre_saved_sgf_data = sio.loadmat(mat_file_path)

    sgfl_specific = pre_saved_sgf_data['sgfl'][0]
    sgfr_specific = pre_saved_sgf_data['sgfr'][0]

    # Round down and round up the minimum and maximum energy values respectively
    Emin = math.floor(-offset_efermi - InputEnergyRange)
    Emax = math.ceil(-offset_efermi + InputEnergyRange)

    # Select points in the energy range
    indices = np.where((specific_energy_points >= Emin) & (specific_energy_points <= Emax))[0]
    selected_energy_points = specific_energy_points[indices]
    prepared_sgfl = [sgfl_specific[i] for i in indices]
    prepared_sgfr = [sgfr_specific[i] for i in indices]

    print(" ")
    print(f"The transport calculation under {eefs:.4f} electric field begins...")

    E_num = math.ceil((Emax - Emin) / InputEnergyInterval)
    Erange = np.round(np.linspace(Emin, Emax, E_num+1), 5)

    # Call the interpolation function
    sgfl_interp_mat, sgfr_interp_mat = interpolate_sgf_matrices(selected_energy_points, prepared_sgfl, prepared_sgfr, Erange)

    Trans = l3_calculate_transmission_eef(Erange, sgfl_interp_mat, sgfr_interp_mat, h_ml, s_ml, h_mr, s_mr, h_m, s_m, save_mat_files)

    # Save transmission file
    transmission_file = os.path.join(output_dir, 'Transmission.txt')
    np.savetxt(transmission_file, np.column_stack((Erange + np.real(phi - efermi), np.abs(Trans))),
               fmt='%10.5f\t%12.8e', delimiter='\t', header='Erange\tTransmission', comments='')

    plot_file = os.path.join(output_dir, 'Transmission.png')
    matplotlib.use('Agg')
    plt.figure()
    # y_fixed = 1.0
    # plt.scatter(eigenvalues, [y_fixed] * len(eigenvalues), edgecolor='b', linewidth=1, s=50)
    plt.plot(Erange + offset_efermi, np.log10(Trans), label='Transmission', color='blue')
    plt.title('Transmission Spectrum')
    plt.xlim([-InputEnergyRange, InputEnergyRange])
    plt.xlabel('Energy (eV)', fontweight='bold')
    plt.ylabel('Transmission (log10)', fontweight='bold')
    plt.legend()
    plt.savefig(plot_file)

    # print(" ")
    # print("Fermi energy (eV): ", -offset_efermi)
    # print(" ")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(" ")
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print(f"The calculation under {eefs:.4f} electric field is completed!")


if __name__ == "__main__":
    main()