import numpy as np
import scipy.io as sio

def find_closest_G_and_Gamma(energy, offset_efermi):
    # Load .mat files
    Gr_data = sio.loadmat('Gr_matrices.mat')
    GammaL_data = sio.loadmat('GammaL_matrices.mat')
    GammaR_data = sio.loadmat('GammaR_matrices.mat')

    # Extract the key name and corresponding energy value
    Gr_keys = [key for key in Gr_data if key.startswith('Gr_E')]
    GammaL_keys = [key for key in GammaL_data if key.startswith('GammaL_E')]
    GammaR_keys = [key for key in GammaR_data if key.startswith('GammaR_E')]

    energies = np.array([float(key.split('_E')[-1]) for key in Gr_keys])

    # Find the index closest to the input energy
    energy_diff = np.abs(energies - energy)
    closest_index = np.argmin(energy_diff)
    closest_energy = energies[closest_index]
    closest_energy += offset_efermi

    # Retrieve matrices corresponding to the closest energy
    Gr_key = Gr_keys[closest_index]
    GammaL_key = GammaL_keys[closest_index]
    GammaR_key = GammaR_keys[closest_index]

    print(" ")
    print(f"Found closest energy: {closest_energy:.5f} eV.")

    Gr = Gr_data[Gr_key]
    GammaL = GammaL_data[GammaL_key]
    GammaR = GammaR_data[GammaR_key]
    return Gr, GammaL, GammaR, closest_energy
