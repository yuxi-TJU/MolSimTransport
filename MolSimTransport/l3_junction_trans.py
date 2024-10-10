'''
A python script that computes the transport of molecular junction containing the principal layer of the electrode under the xTB method.
'''

import os
import numpy as np
import math
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import PchipInterpolator
import matplotlib
import sys
import time

start_time = time.time()

def generate_dftb_input_step1(poscar_file, output_file):
    input_data = f"""
Geometry = vaspFormat {{
    <<< "{poscar_file}"
}}

Hamiltonian = xTB {{
    Method = "GFN1-xTB"
    kPointsAndWeights = {{
        0.    0.    0.       1
    }}
}}
Options = {{
    WriteCharges = Yes
    WriteHS = No
    WriteRealHS = No
}}
Analysis = {{
    WriteBandOut = Yes
}}
"""
    with open(output_file, 'w') as output:
        output.write(input_data)

def generate_dftb_input_step2(poscar_file, output_file):
    input_data = f"""
Geometry = vaspFormat {{
    <<< "{poscar_file}"
}}

Hamiltonian = xTB {{
    Method = "GFN1-xTB"
    kPointsAndWeights = {{
        0.    0.    0.       1
    }}
    ReadInitialCharges = Yes
}}
Options = {{
    WriteHS = Yes
}}
Analysis = {{
    WriteEigenvectors = Yes
    EigenvectorsAsText = Yes
}}
"""
    with open(output_file, 'w') as output:
        output.write(input_data)

def main():
    # Define the path to the POSCAR file, uniform electric field strength and direction
    poscar_file = str(input(">>> Enter POSCAR file name(e.g., coor.POSCAR): "))
    dftb_input_file = "dftb_in.hsd"
    dftb_output_file = "dftb_output.out"

    # Define the energy range
    InputEnergyRange = float(input(">>> Specify the energy range (from 'Fermi energy' plus or minus 'energy range'(The value cannot exceed 4), e.g., 2)(eV): "))

    InputEnergyNumber = int(input(">>> Specify the energy number, (e.g., 401)(int): "))

    # Generate the DFTB+ input file for step 1
    generate_dftb_input_step1(poscar_file, dftb_input_file)

    # Run the DFTB+ program
    os.system("dftb+ > " + dftb_output_file)
    os.rename("dftb_in.hsd", "1st_dftb_in.hsd")
    print("1st DFTB+ calculation done!")

    # Generate the DFTB+ input file for step 2
    generate_dftb_input_step2(poscar_file, dftb_input_file)

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

    dat_file_path = os.environ.get('DAT_FILE_PATH')

    # Load electrode data
    h1_2pl_data = np.loadtxt(os.path.join(dat_file_path, 'h1-kpoint-avg.dat'))
    h1_2pl_kavg = h1_2pl_data[:, 0::2] + 1j * h1_2pl_data[:, 1::2]

    # s1_2pl_data = np.loadtxt(os.path.join(dat_file_path, 's1-kpoint-avg.dat'))
    # s1_2pl_kavg = s1_2pl_data[:, 0::2] + 1j * s1_2pl_data[:, 1::2]

    h2_2pl_data = np.loadtxt(os.path.join(dat_file_path, 'h2-kpoint-avg.dat'))
    h2_2pl_kavg = h2_2pl_data[:, 0::2] + 1j * h2_2pl_data[:, 1::2]

    # s2_2pl_data = np.loadtxt(os.path.join(dat_file_path, 's2-kpoint-avg.dat'))
    # s2_2pl_kavg = s2_2pl_data[:, 0::2] + 1j * s2_2pl_data[:, 1::2]

    efermi = -12.3377

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

    offset_efermi = np.real(phi - efermi)

    # Select specific energy points and interpolations
    specific_energy_points = np.arange(-15, -4, 1)

    data = sio.loadmat(os.path.join(dat_file_path, 'sgf_k10_specific_results_15to5_0.1.mat'))
    sgfl_specific = data['sgfl'][0]
    sgfr_specific = data['sgfr'][0]

    # Round down and round up the minimum and maximum energy values respectively
    Emin = math.floor(-offset_efermi - InputEnergyRange)
    Emax = math.ceil(-offset_efermi + InputEnergyRange)

    # Select points in the energy range
    indices = np.where((specific_energy_points >= Emin) & (specific_energy_points <= Emax))[0]
    selected_energy_points = specific_energy_points[indices]
    prepared_sgfl = [sgfl_specific[i] for i in indices]
    prepared_sgfr = [sgfr_specific[i] for i in indices]

    print("Starting transport calculation...")

    E_num = InputEnergyNumber
    Erange = np.linspace(Emin, Emax, E_num)

    # Prepare the SGF matrix for subsequent interpolation
    sgfl_interp_mat = np.zeros((prepared_sgfl[0].shape[0], prepared_sgfl[0].shape[1], E_num), dtype=complex)
    sgfr_interp_mat = np.zeros((prepared_sgfr[0].shape[0], prepared_sgfr[0].shape[1], E_num), dtype=complex)

    # Interpolate each element of each SGF matrix
    for j in range(len(selected_energy_points)):
        for k in range(prepared_sgfl[j].shape[0]):
            for m in range(prepared_sgfr[j].shape[1]):
                # Extract specific elements
                sgfl_elements = np.array([x[k, m] for x in prepared_sgfl])
                sgfr_elements = np.array([x[k, m] for x in prepared_sgfr])

                # The real and imaginary parts are interpolated separately
                sgfl_real_interp = PchipInterpolator(selected_energy_points, np.real(sgfl_elements))
                sgfl_imag_interp = PchipInterpolator(selected_energy_points, np.imag(sgfl_elements))

                sgfr_real_interp = PchipInterpolator(selected_energy_points, np.real(sgfr_elements))
                sgfr_imag_interp = PchipInterpolator(selected_energy_points, np.imag(sgfr_elements))

                # Combine real and imaginary parts to form a complex number
                sgfl_interp_mat[k, m, :] = sgfl_real_interp(Erange) + 1j * sgfl_imag_interp(Erange)
                sgfr_interp_mat[k, m, :] = sgfr_real_interp(Erange) + 1j * sgfr_imag_interp(Erange)

    # Initialize transmission function
    Trans = np.zeros(len(Erange))

    # Computed transmission function
    for i in range(len(Erange)):
        # Use the interpolated SGF matrix
        sgf_left = np.squeeze(sgfl_interp_mat[:, :, i])
        sgf_right = np.squeeze(sgfr_interp_mat[:, :, i])
        tCL = h_ml - Erange[i] * s_ml
        tLC = tCL.conj().T
        tCR = h_mr - Erange[i] * s_mr
        tRC = tCR.conj().T
        Sigmal = tCL @ sgf_left @ tLC
        Sigmar = tCR @ sgf_right @ tRC

        Gammal = -2 * np.imag(Sigmal)
        Gammar = -2 * np.imag(Sigmar)

        eta = 1e-8
        G = np.linalg.inv((Erange[i] + 1j * eta) * s_m - (h_m + Sigmal + Sigmar))

        T_list = Gammal @ G @ Gammar @ G.conj().T
        Trans[i] = np.real(np.trace(T_list))

    # calculate the eigenvalues
    eigenvalues = np.sort(np.real(eigh(h_m + Sigmal + Sigmar, s_m, eigvals_only=True))) + offset_efermi

    # Save transmission file
    np.savetxt('Transmission.txt', np.column_stack((Erange + np.real(phi - efermi), np.abs(Trans))),
               fmt='%22.15f\t%22.15f', delimiter='\t', header='Erange\tTransmission', comments='')

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
    print(" ")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print("Done!")

if __name__ == "__main__":
    main()