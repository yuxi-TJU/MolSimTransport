import numpy as np
import matplotlib.pyplot as plt
import os

# Define parameters
num_files = 201  # Number of files
base_file_name_pattern = "./{}/Transmission.txt"  # File path pattern
target_row = 172  # Row index to extract data (starting from 1)

# Read the first file to extract the energy range
file_name = base_file_name_pattern.format(0)
data = np.loadtxt(file_name, skiprows=1)  # Skip the first line (header)
energy_range = data[:, 0]
num_energy_points = len(energy_range)

all_data = np.zeros((num_energy_points, num_files))

# Iterate through all files and read transmission data
for k in range(num_files):
    file_name = base_file_name_pattern.format(k)
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} does not exist.")
    data = np.loadtxt(file_name, skiprows=1)
    transmission_data = data[:, 1]
    transmission_data = np.clip(transmission_data, 1e-14, None)
    all_data[:, k] = np.log10(transmission_data)

# Define axes
distance = np.linspace(0, 10, num_files)  # Extension distance (Ang)
target_transmission = all_data[target_row - 1, :]  # Transmission at target row

# Create figure with subplots
plt.figure(figsize=(14, 6))

# Subplot 1: Heatmap
plt.subplot(1, 2, 1)
plt.imshow(all_data, aspect='auto', extent=[0, 10, energy_range[-1], energy_range[0]], cmap='hot', interpolation='bicubic')
plt.colorbar(label="Log10(Transmission)")
plt.xlabel("Extension Distance (Ang)")
plt.ylabel("Energy (eV)")
plt.title("Transmission Heatmap (Log10)")
plt.gca().invert_yaxis()  # Ensure energy range is increasing from bottom to top

# Subplot 2: Line Plot
plt.subplot(1, 2, 2)
plt.plot(distance, target_transmission, marker='o', linewidth=2, label=f"Row {target_row}")
plt.xlabel("Extension Distance (Ang)")
plt.ylabel("Log10(Transmission)")
plt.title(f"Transmission at -10 eV (Log10)")
plt.ylim(-14, -6)
plt.xlim(0, 10)
# plt.legend()

plt.tight_layout()
plt.savefig("transmission_subplots.png", dpi=600)
print("Plot saved as 'transmission_subplots.png'.")
