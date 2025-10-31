# MolSimTransport (MST) L1 Scheme Computation Guide

## 1. L1 Scheme Overview
The L1 scheme is the most fundamental tier in the QDHC framework, focusing on analyzing transport properties dominated by the molecule's intrinsic electronic structure.
 - **System of Study**: Isolated Molecule.
 - **Physical Approximation**: It employs the Wide-Band Approximation (WBA). The influence of the electrodes is simplified to a constant, energy-independent coupling strength (Γ) and self-energy (Σ=−iΓ/2) applied at the molecule's anchoring atoms.
 - **Applicable Problems**: Quantum Interference effects (DQI), e.g., comparing para- vs. meta-linkages; Fano resonances; Effects of molecular conformation, substituents, or charge state on transport.
 - **Available Modules**: MST provides two modules to perform L1 calculations: `L1_EHT` and `L1_XTB`.

## 2. Input Preparation
Calculations under the L1 scheme only require the molecule's structure file.

1. **Structure File**:
 - Molecular coordinates must be provided in `.xyz` format. All structures needed for comparison should be prepared.
2. **Anchor Atom Indices**:
 - The user must check the `.xyz` file in a visualization software (like VESTA, Multiwfn) beforehand to identify the **atom indices** of the anchoring atoms (e.g., S, C) that connect to the left and right electrodes.
3. **Directory Structure**:
 - It is recommended to create a separate directory for each independent calculation task and place the corresponding `.xyz` file inside it.
 - For multi-structure scanning tasks (like stretching), all `.xyz` files can be placed in one directory and processed in batch using a script.

## 3. Computational Workflow
The L1 workflow involves selecting either the `L1_EHT` or `L1_XTB` module, setting the parameters, and executing the calculation.

### 3.1 `L1_EHT` Module
The `L1_EHT` module uses Extended Hückel Theory (EHT) for the electronic structure calculation. It is extremely fast and suitable for simple π-conjugated systems or scenarios requiring scans over many conformations (e.g., stretching).

**Command**:
```Bash
L1_EHT -f [filename] -L [left_indices] -R [right_indices] [options]
```

**Core Parameters**:

| Parameter | Meaning | Default | Example Usage |
| :--- | :--- | :--- | :--- |
| `-f`, `--file` | Specify the input `.xyz` structure file | (None) | `-f 0.xyz` |
| `-L`, `--left` | Specify left anchor atom indices (space-separated) | (None) | `-L 36` |
| `-R`, `--right`| Specify right anchor atom indices (space-separated) | (None) | `-R 78` |
| `-C`, `--coupling`| Specify the coupling strength Γ (eV) | `0.1` | `-C 0.2` |
| `--Erange` | Specify energy range [min max] (eV) | `-15 -6` | `--Erange -12 -8.5` |
| `--Enum` | Specify the number of energy points | `900` | `--Enum 600` |

### 3.2 `L1_XTB` Module
The `L1_XTB` module uses the GFN-xTB method (via the tblite library) for the electronic structure calculation. It is more accurate than EHT and better handles charge distribution, non-neutral systems (ions), and complex electronic structures. It also allows independent set the left and right coupling strengths, as well as specification of the molecule’s total charge.

**Command**:
```Bash
L1_XTB -f [filename] -L [left_indices] -R [right_indices] [options]
```

**Core Parameters**:

| Parameter | Meaning | Default | Example Usage |
| :--- | :--- | :--- | :--- |
| `-f`, `--file` | Specify the input `.xyz` structure file | (None) | `-f AQ.xyz` |
| `-L`, `--left` | Specify left anchor atom indices (space-separated) | (None) | `-L 1` |
| `-R`, `--right`| Specify right anchor atom indices (space-separated) | (None) | `-R 2` |
| `-C`, `--coupling`| Specify the coupling strength Γ (eV) | `1` | `-C 0.2` |
| `--CL` | Configure the left coupling individually (overrides -C) | (None) | |
| `--CR` | Configure the right coupling individually (overrides -C) | (None) | |
| `-m`, `--method`| Calculation method: 1 = GFN1-xTB, 2 = GFN2-xTB | `1` | `-m 1` |
| `--Erange` | Energy range (eV). **Note**: This is E_F ± this value. | `4.0` | `--Erange 2` |
| `--Enum` | Specify the number of energy points | `800` | `--Enum 1000` |
| `--charge` | Total charge of the molecule (float) | `0.0` | `--charge -1` |

**Note**: The `L1_XTB` module automatically defines the "Fermi level" (E_F) as the average of HOMO and LUMO. The `--Erange` parameter is relative to this E_F.

## 4. Output

### 4.1 Main Output Files
 - `Transmission.png`: A log-scale plot of the transmission spectrum.
 - `Transmission.txt`: The raw data of the transmission spectrum (linear scale).

### 4.2 Screen/Log Output
 - At the start, L1_XTB will print the calculation method (e.g., GFN1-xTB) and total orbital count.
 - On completion, both modules will print the HOMO and LUMO orbital indices and their energies (eV). 
 - L1_XTB will also print the calculated "Fermi energy" (eV).

---

# MolSimTransport (MST) L2 Scheme Computation Guide

## 1. L2 Scheme Overview
The L2 scheme is the intermediate tier in the QDHC framework, aimed at analyzing interface-dominated transport using an explicitly modeled "extended molecule + electrode clusters" system that captures local contact effects.
 - **System of Study**: "Extended Molecule + Electrode Clusters". The system explicitly includes the molecule plus the electrode atoms forming the interface. The extended molecule consists of the central molecule connected to 1, 3, or 4 Au atoms on each side, corresponding to the adatom, trimer, and pyramid configurations, respectively.
 - **Physical Approximation**: This scheme computes the Hamiltonian for the entire "Extended Molecule (EM) + Cluster" system . The electrode self-energy (Σ) is derived from the coupling matrices obtained via partitioning, combined with a constant local density of states (LDOS) approximation for the bulk electrode's Green's function.
 - **Applicable Problems**: Effects of contact geometry on coupling and conductance; Influence of anchoring chemistry (e.g., sp vs. sp3 hybridization) on resonance broadening; Transport through non-traditional binding motifs.
 - **Available Modules**: MST provides two modules for the L2 scheme: `L2_Align` (for building the full system) and `L2_Trans` (for calculation).

## 2. Input Preparation
Calculations under the L2 scheme require a user-provided Extended Molecule (EM) structure file, which is built from MST's templates.

1. **MST Templates (Provided)**:
 - MST provides two EM templates (Both sides use trimer (`3au-em.xyz`) or pyramid (`4au-em.xyz`) Au configurations) and a cluster template (`SuppliedCluster.xyz`) in the `[MST_root]/share/em/` directory.
2. **User-Created EM File**:
 - The user must **manually** create an "Extended Molecule" `.xyz` file. This is done by selecting an EM template, replacing the placeholder molecule with the target molecule, and adjusting the geometry as needed (e.g., setting tilt angles).
 - It is essential that the Au atoms **on both sides** of the EM template remain **a rigid, unified block**, with **their original order in the template `.xyz` file strictly maintained**.
 - A set of these EM files must be prepared for the systems being compared.
3. **Directory Structure**:
 - It is recommended to create a separate directory for each EM system to be calculated(e.g., `project_root/em1/`, `project_root/em2/`, etc.).
 - Place the corresponding user-created EM file (e.g., `em1.xyz`) inside its directory.

## 3. Computational Workflow
The L2 workflow is a two-step process involving: (1) system construction with `L2_Align` and (2) transport calculation with `L2_Trans`.

### 3.1 Step 1: `L2_Align` Module (System Construction)
This interactive module builds the final "EM + Cluster" system (named `aligned.xyz`) by combining your user-created EM file with the MST-provided `SuppliedCluster.xyz` template (This Cluster file does not need to be present in the working directory).

**Command**:
```Bash
L2_Align
```

**Process**:
1. Run the command in the subdirectory (e.g., `project_root/em1/`).
2. When prompted, enter the name of your user-created EM file (e.g., `em1.xyz`).
3. The module will automatically generate the `aligned.xyz` file in the same directory.

### 3.2 Step 2: `L2_Trans` Module (Transport Calculation)
This interactive module calculates the transmission spectrum for the `aligned.xyz` system. After specifying the system name and computational method, the program first executes an electronic structure calculation. It then prompts the user to define the cluster size (in atoms), the energy range, and the number of energy points before initiating the transport calculation.

**Command**:
```Bash
L2_Trans
```

**Core Interactive Prompts**:

| Prompt | Meaning | Default | Example Usage |
| :--- | :--- | :--- | :--- |
| `Enter XYZ file name (...)` | Specify the system file to calculate. | (None) | `aligned.xyz` |
| `Enter calculation method (...)` | Electronic structure method: 1 = GFN1-xTB, 2 = GFN2-xTB. | (None) | `1` |
| `Specify the cluster atom number (25 or 28)`| Defines how to partition the EM and cluster. | (None) | `25` |
| `Specify the energy range (...)` | Energy range (eV). **Note**: This is E_F ± this value. | (None) | `2` |
| `Specify the energy interval (...)` | Energy step size (eV) for the spectrum. | (None) | `0.01` |

**Note**: The `L2_Trans` module automatically defines the "Fermi level" (E_F) as the HOMO energy of the isolated Au cluster (e.g., the 25-atom/28-atom cluster), The energy range is relative to this E_F; About "cluster atom number" parameter, set 28 only for adatom interfaces; otherwise, keep the default 25.

## 4. Output

### 4.1 Main Output Files
 - `Transmission.png`: A log-scale plot of the transmission spectrum.
 - `Transmission.txt`: The raw data of the transmission spectrum (linear scale).

### 4.2 Screen/Log Output
 - The module will first show the `tblite` (GFN-xTB) SCF convergence cycles.
 - After the transport calculation, it will print the calculated "Fermi energy" (i.e., the cluster's HOMO energy) in eV.

---

# MolSimTransport (MST) L3 Scheme Computation Guide

## 1. L3 Scheme Overview
The L3 scheme is the most advanced tier in the QDHC framework, targeting problems dominated by level alignment or bias effects, and offering tools to analyze MPSH orbitals and transmission eigenstates.

 - **System of Study**: "Full Molecular Junction". The system for the electronic structure calculation consists of the Extended Molecule (EM) connected to the Principal Layers (PL) of the electrodes.
 - **Physical Approximation**: This scheme uses pre-calculated, energy-dependent Surface Green's Functions (SGF) for the bulk electrodes, which are loaded and interpolated during the transport calculation. This allows for a proper alignment of molecular orbitals to the electrode's Fermi level (E_F).
 - **Finite-Bias**: Non-equilibrium transport (I-V curves) is simulated by applying a Uniform External Electric Field (EEF) across the Extended Molecule region during the electronic structure calculation.
 - **Applicable Problems**: Determining the dominant transport channel by aligning the transmission spectrum to the electrode E_F. Calculating I-V characteristics, rectification ratios, and other finite-bias phenomena.
 - **Available Modules**: `L3_Trans` (zero-bias), `L3_EEF` (finite-bias, called by script), `L3_MPSH` (analysis), `L3_EC` (analysis).

## 2. Input Preparation
The L3 scheme requires a full junction structure file, which must be converted to `POSCAR` format.

1. **MST Templates (Provided)**:
 - MST provides full junction `.xyz` templates in the `[MST_root]/share/device/` directory, featuring different interfaces (e.g., `junction_example_adatom_amine.xyz`, `junction_trimer_...`, corresponding to the adatom, trimer, and pyramid configurations).
2. **User-Created Junction File (`.xyz`)**:
 - The user must manually create a junction `.xyz` file by replacing the placeholder molecule in an MST template with their target molecule (e.g., creating `junction_adatom_m1.xyz`).
3. **Convert Script: `xyz2POSCAR.py`**:
 - This Python script (located in `[MST_root]/share/`) is required. It converts the user-created junction `.xyz` file into the `POSCAR` format that DFTB+ (the L3 electronic structure engine) understands.
4. **Additional Input for Finite-Bias**:
 - To run finite-bias (`L3_EEF`) calculations, the user must also measure and record the length of the extended molecule (in Ångstroms) along the transport (z) direction.
5. **Directory Structure**:
 - Create a separate directory for each junction system (e.g., `project_root/j1/`, `project_root/j2/`).
 - Place the corresponding user-created junction `.xyz` file inside its directory.

## 3. Computational Workflow
The L3 workflow is a multi-step process and may involve different modules, depending on the specific requirements. All L3 calculations must begin with Step 1.

### 3.1 Step 1: File Conversion
`xyz2POSCAR.py` script must be run first to prepare the input file for the L3 modules.

**Process**:
1. Copy `xyz2POSCAR.py` from the `[MST_root]/share/` directory into the working directory (e.g., `project_root/j1/`).
2. Edit the script: Open `xyz2POSCAR.py` and change the `xyz_filename` variable to match the user-created junction file (e.g., `xyz_filename = 'junction_adatom_m1.xyz'`).
3. Run the script:
```Bash
python xyz2POSCAR.py
```
4. Outputs: This script generates two files in working directory:
 - `POSCAR`: The input structure file for DFTB+.
 - `EM_atom.txt`: Defines the atoms belonging to the EM, required for post-processing.

### 3.2 Step 2(Option A): `L3_Trans` (Zero-Bias Calculation)
Use this interactive module to calculate the zero-bias transmission spectrum. After specifying the POSCAR file name, energy range, and energy interval as prompted, the module invokes `DFTB+` to carry out the electronic structure calculation, then proceeds to the transport calculation.

**Command**:
```Bash
L3_Trans
```
**Core Interactive Prompts**:

| Prompt | Meaning | Default | Example Usage |
| :--- | :--- | :--- | :--- |
| `Enter POSCAR file name(...)` | Specify the `POSCAR` file to calculate. | (None) | `POSCAR` |
| `Specify the energy range (...)` | Energy range (eV) E_F ± this value. **Max 4.0**. | (None) | `2` |
| `Specify the energy interval (...)` | Energy step size (eV) for the spectrum. | (None) | `0.01` |


### 3.3 Step 2(Option B): `L3_EEF` (Finite-Bias Calculation)
This workflow is used to calculate non-equilibrium transport, such as transmission spectrum under bias and I-V curves. The `L3_EEF` module simulates the effect of a bias voltage by applying a finite Uniform External Electric Field (EEF) during the `DFTB+` electronic structure calculation. The key concept is:

 - **Field vs. Voltage**: Voltage (V) is not input directly. Instead, an electric field strength (E) is input in atomic units (a.u.).
 - **Length (L_EM)**: The length of the Extended Molecule (EM) along the transport direction must be manually measured (in Ångstroms).
 - **Conversion**: The script calculates the corresponding voltage using the formula **V=E×L_EM**.  
 (Unit conversion: **1 a.u. of field is 5.142×10^{11} V/m**; **1 Å is 1×10^{−10} m**).

`L3_EEF` is not run directly. Instead, one of the helper scripts provided in the `[MST_root]/share/` directory is used:
 - `current_serial.py`: Runs each bias point one by one.
 - `current_parallel.py`: Runs multiple bias points in parallel.

**Process**:

1. **Copy Script**: Copy `current_parallel.py` (or `current_serial.py`) from the `[MST_root]/share/` directory into the working directory (e.g., `project_root/j1/`).
2. **Edit the script**: Open `current_parallel.py` and modify the following variables located in the `main` function:

| Variable in Script | Meaning | Example Value |
| :--- | :--- | :--- |
| `poscar_file` | The name of the `POSCAR` file. | `"POSCAR"` |
| `Length` | The EM length (Å) you measured. | `26.67857` |
| `input_energy_range` | Energy range (eV), $E_F \pm$ this value. | `2` |
| `input_energy_interval` | Energy step size (eV). | `0.0025` |
| `electric_field_range` | Numpy array of field strengths (a.u.). | `np.arange(-0.0008, 0.0009, 0.0001)` |
| `max_workers` | Number of parallel calculations. | `2` |

*Note1: The `np.arange(-0.0008, 0.0009, 0.0001)` will calculate 16 field points ranging from –0.0008 to +0.0008 a.u., excluding the zero-field point.*  
*Note2: Use the line `electric_field_range = np.array([0.0006])` to calculate transport at a single electric field/bias point.*

3. **Run the script**:
```bash
python current_parallel.py
```
4. **Execution Result**: The script will automatically:
 - Create a new directory for each bias point (each value in `electric_field_range`).
 - Run the `L3_EEF` calculation (including `DFTB+` with the EEF) inside each directory.
 - After all jobs are finished, it collects the transmission data from every directory and merges them into `combined_transmission.txt`.
 - Finally, it integrates the transmission spectra to calculate the current at each bias, saving the final I-V data to `voltage_current.txt`.

**Memory Warning**: Parallel calculations (high `max_workers`) combined with a dense energy grid (small `input_energy_interval`) consume a large amount of RAM. If your jobs are crashing, reduce `max_workers` or use `current_serial.py`.

## 4. Output and Analysis

### 4.1 `L3_Trans` (Zero-Bias) Output
 - **Main Files**: DFTB+ I/O files, `Transmission.png`, `Transmission.txt`. In both transmission files, the Fermi level is shifted to 0 eV.
 - **Analysis Files**: `Gr_matrices.mat`, `GammaL_matrices.mat`, `GammaR_matrices.mat`, `mpsh_eigenvalues.txt`, `mpsh_eigenvectors.txt`. These are used by `L3_MPSH` and `L3_EC`.
 - **Screen Output**: Prints the calculated Fermi energy (eV).

### 4.2 `L3_EEF` (Finite-Bias) Output
 - **Main Files**: 
	- `combined_transmission.txt`: Merged transmission data for all biases. 
	- `voltage_current.txt`: The final I-V curve data. This file stores the voltage (converted from the applied electric field) and the corresponding current.

### 4.3 Post-Analysis Modules (for `L3_Trans` results)
*Note: These two modules are mainly used to analyze results under zero bias. To analyze bias-dependent cases, set `save_mat_files` to `True` in the `current_parallel.py` script.*

#### 4.3.1 `L3_MPSH` Module
This module generates a file for visualizing the Molecular Projected Self-Consistent Hamiltonian (MPSH) orbitals, which are the basis orbitals of the EM.
 - **Command**: `L3_MPSH` (no arguments).
 - **Action**: Uses `EM_atom.txt` and `mpsh_eigenvectors` to generate `MPSH.molden` for visualizing the EM orbitals(can be opened in visualization software like Multiwfn).

#### 4.3.2 `L3_EC` Module
This module analyzes the the eigenchannel at a specific energy and projects it onto the MPSH orbitals to identify its origin.
 - **Command**: `L3_EC` (interactive).
 - **Action**: Prompts for an energy value (e.g., `-1.82`). It then uses the `.mat` files to project the transport eigenchannel at that energy onto the MPSH orbitals.
 - **Output**:
	- **Screen Output**: Lists MPSH orbitals and their percentage contribution (weight) to transport at the chosen energy, identifying the dominant orbital (e.g., "orbital 49: 0.88...").
	- **Eigenchannel Molden Files**: Generates three `.molden` files (corresponding to the absolute value, real part, and imaginary part) for the dominant eigenchannel to visualize its spatial distribution(e.g., `EigenChannel_abs_-1.82070.molden`).

#### 4.3.3 Summary Analysis Workflow
In a complete L3 zero-bias workflow, the `L3_Trans` module computes the transmission spectrum, and the `L3_EC` module identifies the MPSH orbitals that dominate the selected transmission peak. The `L3_MPSH` module then generates a Molden file for comparison with orbitals of the isolated molecule (provided separately using the same electronic structure level) to determine which molecular orbital or electrode atoms contributes to the transmission feature. It should be noted that this complete workflow is not required for every calculation; the analysis modules should be used only when such analyses are explicitly needed.
