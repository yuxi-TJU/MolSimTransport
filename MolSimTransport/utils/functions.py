import numpy as np
import scipy.linalg as la
import scipy.io as sio
from scipy.linalg import eigh
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed

#######################################
############  L1-FUNCTION  ############
####################################### 

def build_coupling_matrix(indexes, wba, N, dtype=float):
    matrix = np.zeros((N, N), dtype=dtype)
    for start, end in indexes:
        matrix[start:end, start:end] = np.diag([wba] * (end - start))
    return matrix

def l1_calculate_transmission(Erange, S, H, left_indexes, right_indexes, GammaL_wba, GammaR_wba, z_plus=1e-13j):
    E_num = len(Erange)
    Trans = np.zeros(E_num, dtype=complex)
    
    for i in range(E_num):
        GammaL = build_coupling_matrix(left_indexes, GammaL_wba, H.shape[0])
        GammaR = build_coupling_matrix(right_indexes, GammaR_wba, H.shape[0])
        SigmaL = build_coupling_matrix(left_indexes, -0.5j * GammaL_wba, H.shape[0], dtype=complex)
        SigmaR = build_coupling_matrix(right_indexes, -0.5j * GammaR_wba, H.shape[0], dtype=complex)

        G = np.linalg.inv((Erange[i] + z_plus) * S - H - SigmaL - SigmaR)
        Trans_list = G @ GammaL @ G.T.conj() @ GammaR
        Trans[i] = np.real(np.trace(Trans_list))
        
    return Trans

#######################################
############  L2-FUNCTION  ############
####################################### 

def l2_calculate_transmission(H_full, S_full, Erange, N_L, N_R):
    LDOS=0.036 
    z_plus=1e-13j

    N_m = H_full.shape[0] - N_L - N_R 

    H_m = H_full[N_L:N_L+N_m, N_L:N_L+N_m]
    V_Lm = H_full[:N_L, N_L:N_L+N_m]
    V_mL = H_full[N_L:N_L+N_m, :N_L]
    V_Rm = H_full[N_L+N_m:, N_L:N_L+N_m]
    V_mR = H_full[N_L:N_L+N_m, N_L+N_m:]
    S_m = S_full[N_L:N_L+N_m, N_L:N_L+N_m]
    S_Lm = S_full[:N_L, N_L:N_L+N_m]
    S_mL = S_full[N_L:N_L+N_m, :N_L]
    S_Rm = S_full[N_L+N_m:, N_L:N_L+N_m]
    S_mR = S_full[N_L:N_L+N_m, N_L+N_m:]

    mpsh_w, mpsh_v = eigh(H_m, S_m)

    with open("mpsh_eigenvalues.txt", "w") as f:
        f.write("# mpsh_eigenvalues\n")
        np.savetxt(f, mpsh_w, fmt='%10.5f')

    with open("mpsh_eigenvectors.txt", "w") as f:
        f.write("# mpsh_eigenvectors\n")
        np.savetxt(f, mpsh_v, fmt='%22.16f') 

    g_L = -1j * np.pi * LDOS * np.eye(N_L)
    g_R = -1j * np.pi * LDOS * np.eye(N_R)
    Trans = np.zeros(len(Erange))

    for i, EE in enumerate(Erange):
        Sigma_L = (EE * S_mL - V_mL) @ g_L @ (EE * S_Lm - V_Lm)
        Sigma_R = (EE * S_mR - V_mR) @ g_R @ (EE * S_Rm - V_Rm)
        Gamma_L = 1j * (Sigma_L - Sigma_L.T.conj())
        Gamma_R = 1j * (Sigma_R - Sigma_R.T.conj())

        G = la.inv((EE + z_plus) * S_m - H_m - Sigma_L - Sigma_R)
        Trans[i] = np.real(np.trace(G @ Gamma_L @ G.T.conj() @ Gamma_R))
    
    return Trans

#######################################
############  L3-FUNCTION  ############
####################################### 

def generate_1st_dftb_input(poscar_file, output_file):
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

def generate_2nd_dftb_input(poscar_file, output_file):
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

def interpolate_sgf_matrices(selected_energy_points, prepared_sgfl, prepared_sgfr, Erange):
    # Initialize the interpolated SGF matrices
    sgfl_interp_mat = np.zeros((prepared_sgfl[0].shape[0], prepared_sgfl[0].shape[1], len(Erange)), dtype=complex)
    sgfr_interp_mat = np.zeros((prepared_sgfr[0].shape[0], prepared_sgfr[0].shape[1], len(Erange)), dtype=complex)

    # Perform interpolation for each element in SGF matrices
    for k in range(prepared_sgfl[0].shape[0]):
        for m in range(prepared_sgfl[0].shape[1]):
            # Extract elements for each energy point
            sgfl_elements = np.array([x[k, m] for x in prepared_sgfl])
            sgfr_elements = np.array([x[k, m] for x in prepared_sgfr])

            # Interpolate real and imaginary parts separately
            sgfl_real_interp = PchipInterpolator(selected_energy_points, np.real(sgfl_elements))
            sgfl_imag_interp = PchipInterpolator(selected_energy_points, np.imag(sgfl_elements))

            sgfr_real_interp = PchipInterpolator(selected_energy_points, np.real(sgfr_elements))
            sgfr_imag_interp = PchipInterpolator(selected_energy_points, np.imag(sgfr_elements))

            # Combine real and imaginary parts to form complex interpolated values
            sgfl_interp_mat[k, m, :] = sgfl_real_interp(Erange) + 1j * sgfl_imag_interp(Erange)
            sgfr_interp_mat[k, m, :] = sgfr_real_interp(Erange) + 1j * sgfr_imag_interp(Erange)

    return sgfl_interp_mat, sgfr_interp_mat

def save_gamma_matrices_mat(energy, Gr, GammaL, GammaR, Gr_dict, GammaL_dict, GammaR_dict):
    # Save the Gr and gamma matrix of the current energy point to the dictionary
    Gr_dict[f'Gr_E{energy:.4f}'] = Gr
    GammaL_dict[f'GammaL_E{energy:.4f}'] = GammaL
    GammaR_dict[f'GammaR_E{energy:.4f}'] = GammaR

##########################################################

def calc_trans_at_energy(E, sgf_left, sgf_right, h_ml, s_ml, h_mr, s_mr, h_m, s_m, eta=1e-8):
    tCL = h_ml - E * s_ml
    tLC = tCL.conj().T
    tCR = h_mr - E * s_mr
    tRC = tCR.conj().T

    Sigma_L = tCL @ sgf_left @ tLC
    Sigma_R = tCR @ sgf_right @ tRC

    Gamma_L = -2 * np.imag(Sigma_L)
    Gamma_R = -2 * np.imag(Sigma_R)

    h_eff = h_m + Sigma_L + Sigma_R
    Gr = np.linalg.inv((E + 1j * eta) * s_m - h_eff)

    T_matrix = Gamma_L @ Gr @ Gamma_R @ Gr.conj().T
    T = np.real(np.trace(T_matrix))

    return E, T, Gr, Gamma_L, Gamma_R


def l3_calculate_transmission_parallel(Erange, sgfl_interp_mat, sgfr_interp_mat, h_ml, s_ml, h_mr, s_mr, h_m, s_m, output_path=''):
    Trans = np.zeros(len(Erange))
    Gr_dict = {}
    GammaL_dict = {}
    GammaR_dict = {}

    def task(i):
        E = Erange[i]
        sgf_left = np.squeeze(sgfl_interp_mat[:, :, i])
        sgf_right = np.squeeze(sgfr_interp_mat[:, :, i])
        return calc_trans_at_energy(E, sgf_left, sgf_right, h_ml, s_ml, h_mr, s_mr, h_m, s_m)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(task, i) for i in range(len(Erange))]

        for future in as_completed(futures):
            E, T, Gr, GammaL, GammaR = future.result()
            idx = np.where(np.isclose(Erange, E))[0][0]
            Trans[idx] = T

            Gr_dict[f'Gr_E{E:.4f}'] = Gr
            GammaL_dict[f'GammaL_E{E:.4f}'] = GammaL
            GammaR_dict[f'GammaR_E{E:.4f}'] = GammaR

    # save matrix files
    sio.savemat(f'{output_path}Gr_matrices.mat', Gr_dict)
    sio.savemat(f'{output_path}GammaL_matrices.mat', GammaL_dict)
    sio.savemat(f'{output_path}GammaR_matrices.mat', GammaR_dict)

    return Trans


##########################################################
def l3_calculate_transmission(Erange, sgfl_interp_mat, sgfr_interp_mat, h_ml, s_ml, h_mr, s_mr, h_m, s_m, output_path=''):
    Trans = np.zeros(len(Erange))
    Gr_dict = {}
    GammaL_dict = {}
    GammaR_dict = {}

    for i in range(len(Erange)):
        # Use the interpolated SGF matrix
        sgf_left = np.squeeze(sgfl_interp_mat[:, :, i])
        sgf_right = np.squeeze(sgfr_interp_mat[:, :, i])
        tCL = h_ml - Erange[i] * s_ml
        tLC = tCL.conj().T
        tCR = h_mr - Erange[i] * s_mr
        tRC = tCR.conj().T
        Sigma_L = tCL @ sgf_left @ tLC
        Sigma_R = tCR @ sgf_right @ tRC

        Gamma_L = -2 * np.imag(Sigma_L)
        Gamma_R = -2 * np.imag(Sigma_R)

        h_eff = h_m + Sigma_L + Sigma_R

        eta = 1e-8
        Gr = np.linalg.inv((Erange[i] + 1j * eta) * s_m - h_eff)

        # Save Gamma and Green's function matrices
        save_gamma_matrices_mat(Erange[i], Gr, Gamma_L, Gamma_R, Gr_dict, GammaL_dict, GammaR_dict)

        T_list = Gamma_L @ Gr @ Gamma_R @ Gr.conj().T
        Trans[i] = np.real(np.trace(T_list))

    # Save matrices to .mat files
    sio.savemat(f'{output_path}Gr_matrices.mat', Gr_dict)
    sio.savemat(f'{output_path}GammaL_matrices.mat', GammaL_dict)
    sio.savemat(f'{output_path}GammaR_matrices.mat', GammaR_dict)

    return Trans


#######################################
##########  L3-EEF-FUNCTION  ##########
#######################################


def generate_1st_dftb_eef(poscar_file, eefs, eefd, output_file):
    input_data = f"""
Geometry = vaspFormat {{
    <<< "{poscar_file}"
}}

Hamiltonian = xTB {{
  Method = "GFN1-xTB"
  kPointsAndWeights = {{
       0.    0.    0.       1
  }}
  ElectricField = {{
   External {{
    Strength[au] = {abs(eefs)}
    Direction =  0  0  {eefd}  
}}
}}
  Mixer = Broyden {{
    MixingParameter = 0.05000000000000001
  }}
  MaxSCCIterations = 200
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



def generate_2nd_dftb_eef(poscar_file, output_file):
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
    with open(output_file, 'w') as output:
        output.write(input_data)


def l3_calculate_transmission_eef(Erange, sgfl_interp_mat, sgfr_interp_mat, h_ml, s_ml, h_mr, s_mr, h_m, s_m, save_files, output_path=''):
    Trans = np.zeros(len(Erange))
    Gr_dict = {}
    GammaL_dict = {}
    GammaR_dict = {}

    for i in range(len(Erange)):
        # Use the interpolated SGF matrix
        sgf_left = np.squeeze(sgfl_interp_mat[:, :, i])
        sgf_right = np.squeeze(sgfr_interp_mat[:, :, i])
        tCL = h_ml - Erange[i] * s_ml
        tLC = tCL.conj().T
        tCR = h_mr - Erange[i] * s_mr
        tRC = tCR.conj().T
        Sigma_L = tCL @ sgf_left @ tLC
        Sigma_R = tCR @ sgf_right @ tRC

        Gamma_L = -2 * np.imag(Sigma_L)
        Gamma_R = -2 * np.imag(Sigma_R)

        h_eff = h_m + Sigma_L + Sigma_R

        eta = 1e-8
        Gr = np.linalg.inv((Erange[i] + 1j * eta) * s_m - h_eff)

        T_list = Gamma_L @ Gr @ Gamma_R @ Gr.conj().T
        Trans[i] = np.real(np.trace(T_list))

        if save_files:
            save_gamma_matrices_mat(Erange[i], Gr, Gamma_L, Gamma_R, Gr_dict, GammaL_dict, GammaR_dict)

    if save_files:
        sio.savemat(f'{output_path}Gr_matrices.mat', Gr_dict)
        sio.savemat(f'{output_path}GammaL_matrices.mat', GammaL_dict)
        sio.savemat(f'{output_path}GammaR_matrices.mat', GammaR_dict)

    return Trans
