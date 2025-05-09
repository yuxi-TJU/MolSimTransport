'''
A python script computes the transport of molecular junction containing the principal layer of the electrode under the xTB method.
'''

import os
import numpy as np
import math
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib
from importlib.resources import files
from MolSimTransport.utils.functions import generate_1st_dftb_input, generate_2nd_dftb_input, interpolate_sgf_matrices, l3_calculate_transmission
import time

start_time = time.time()

def main():
    # Define the path to the POSCAR file, uniform electric field strength and direction
    poscar_file = str(input(">>> Enter POSCAR file name(e.g., coor.POSCAR): "))
    dftb_input_file = "dftb_in.hsd"
    dftb_output_file = "dftb_output.out"
    
    # Choosing electrode type
    elec_type = input(">>> Choose electrode type (s / l): ").strip().lower()

    if elec_type == "l":
        share_dir = 'share.l3_elec_large'
        efermi = -12.0666
    elif elec_type == "s":
        share_dir = 'share.l3_elec_small'
        efermi = -12.3777
    else:
        raise ValueError("Invalid input! Please enter 's' or 'l'.")

    # Define the energy range
    InputEnergyRange = float(input(">>> Specify the energy range (from 'Fermi energy' plus or minus 'energy range', The value cannot exceed 4 eV, e.g., 2)(float): "))

    InputEnergyInterval = float(input(">>> Specify the energy interval, (e.g., 0.01)(eV)(float): "))

    # Generate the DFTB+ input file for step 1
    generate_1st_dftb_input(poscar_file, dftb_input_file)

    # Run the DFTB+ program
    os.system("dftb+ > " + dftb_output_file)
    os.rename("dftb_in.hsd", "1st_dftb_in.hsd")
    print("1st DFTB+ calculation done!")

    # Generate the DFTB+ input file for step 2
    generate_2nd_dftb_input(poscar_file, dftb_input_file)

    # Run the DFTB+ program
    os.system("dftb+ > " + dftb_output_file)

    # Rename the hamsqr1.dat and oversqr.dat files
    if os.path.exists("hamsqr1.dat"):
        os.rename("hamsqr1.dat", "device_h.dat")
    else:
        print("file not found: hamsqr1.dat")

    if os.path.exists("oversqr.dat"):
        os.rename("oversqr.dat", "device_s.dat")
    else:
        print("file not found: oversqr.dat")
    
    os.rename("dftb_in.hsd", "2nd_dftb_in.hsd")
    print("2nd DFTB+ calculation done!")

    t1 = time.time()
    elapsed_time1 = (t1 - start_time) / 60
    print(f"DFTB+ calculation executed in {elapsed_time1:.2f} minutes.")

    # dat_file_path = os.environ.get('DAT_FILE_PATH')

    # Load electrode data
    h1_file_path = files(share_dir).joinpath('h1-kpoint-avg.dat')
    h1_2pl_data = np.loadtxt(h1_file_path)
    h1_2pl_kavg = h1_2pl_data[:, 0::2] + 1j * h1_2pl_data[:, 1::2]

    h2_file_path = files(share_dir).joinpath('h2-kpoint-avg.dat')
    h2_2pl_data = np.loadtxt(h2_file_path)
    h2_2pl_kavg = h2_2pl_data[:, 0::2] + 1j * h2_2pl_data[:, 1::2]

    # Load the complete system data
    h = 27.2114 * np.loadtxt('device_h.dat', skiprows=5)
    s = np.loadtxt('device_s.dat', skiprows=5)

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

    mat_file_path = files(share_dir).joinpath('sgf_k10_elec_15to5_0.1.mat')
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

    E_num = math.ceil((Emax - Emin) / InputEnergyInterval)
    Erange = np.round(np.linspace(Emin, Emax, E_num+1), 5)

    # Call the interpolation function
    print("Starting interpolation calculation...")
    sgfl_interp_mat, sgfr_interp_mat = interpolate_sgf_matrices(selected_energy_points, prepared_sgfl, prepared_sgfr, Erange)
    t2 = time.time()
    elapsed_time2 = (t2 - start_time) / 60
    print(f"Interpolation calculation executed in {elapsed_time2:.2f} minutes.")

    print("Starting transmission calculation...")
    Trans = l3_calculate_transmission(Erange, sgfl_interp_mat, sgfr_interp_mat, h_ml, s_ml, h_mr, s_mr, h_m, s_m)
    # Trans = l3_calculate_transmission_parallel(Erange, sgfl_interp_mat, sgfr_interp_mat,h_ml, s_ml, h_mr, s_mr, h_m, s_m)
    t3 = time.time()
    elapsed_time3 = (t3 - start_time) / 60
    print(f"Transmission calculation executed in {elapsed_time3:.2f} minutes.")

    # Save transmission file
    np.savetxt('Transmission.txt', np.column_stack((Erange + offset_efermi, np.abs(Trans))),
               fmt='%10.5f\t%12.8e', delimiter='\t', header='Erange\tTransmission', comments='')

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
    plt.savefig('Transmission.png')

    print(" ")
    print("Fermi energy (eV): ", -offset_efermi)
    print(f"Using electrode data from: {share_dir}")
    print(" ")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print("Done!")

if __name__ == "__main__":
    main()