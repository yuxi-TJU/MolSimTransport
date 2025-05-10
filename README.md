<p align="center">
  <img src="https://img.shields.io/badge/release-v1.0.3-orange" alt="release">
  <a href="https://pubs.acs.org/doi/10.1021/acs.jctc.4c01711">
    <img src="https://img.shields.io/badge/DOI-10.1021/acs.jctc.4c01711-blue" alt="DOI">
  </a>
</p>

# **MolSimTransport: A Python package designed for the efficient calculation of transport properties in molecular junctions**

### **v1.0.3 Release Notes**

**L1_XTB Module**: Added the ability to modify the coupling strength of the left and right electrodes separately using the `--CL` and `--CR` options (`-C` sets both simultaneously).

**L3 Scheme**: Provided larger surface Green's function files for electrodes; users can select the appropriate electrode based on their needs

### **Download link**

 `v1.0.3`  [Google Drive Download Link](https://drive.google.com/drive/folders/1jt-2xSUPCP0NKvlZcofwm0wde2g77XUm?usp=sharing)
 `v1.0.2`  [OneDrive Download Link](https://1drv.ms/u/c/8ba50a0504e6a517/ER87VtJizbZMu8WJ0pdjCi4BKi4xc53XNm0uokhEmnEb9A?download=1)

**After the software package is extracted, extract the two electrode files into `~/share/l3_elec_large` and `~/share/l3_elec_small` respectively.**

## **Introduction**
_**MolSimTransport**_, a highly efficient computational scheme within the **Question-Driven Hierarchical Computation (QDHC)** framework, has been developed by [Professor Yu Xi's research group](http://science.tju.edu.cn/info/1124/1632.htm) at Tianjin University. This scheme integrates several transport models across various levels of approximation, complemented by computational methods of different accuracies. It provides a comprehensive, multi-level solution for analyzing the charge transport properties in molecular junctions.
![strategy](https://github.com/user-attachments/assets/6a5232da-9a3a-4e64-ac7a-145b7950ebff)

The QDHC design in _**MolSim-Transport**_ bridges the theoretical gap between DFT+NEGF calculations on the full molecular devices and and simplified theoretical models, by encompassing a range of simplifications and approximations at various levels. Considering that the charge transport properties of molecular devices are jointly determined by the molecule, electrodes, and their interfacial interactions, the calculations accordingly span multiple physical scales from atoms to devices. Driven by the actual researching questions, this method selectively disregards minor factors being less impactful and only focuses on the primary ones that critically influence system behavior. Moreover, by effectively applying theoretical methods at different levels of approximation and precision, this strategy achieves greater efficiency while maintaining accuracy, thus enabling more effective exploration of charge transport behaviors in molecular junction systems.

## **Detailed Description of the QDHC Strategy**
The core of the QDHC (Question-Driven Hierarchical Computation) strategy lies in the refined stratification and approximation strategies at two critical levels: defining the computational system scope and selecting computational methods. Considering that a typical molecular device consists of three parts: the molecule, the molecule-electrode interface, and the source/drain electrodes, the QDHC strategy requires researchers to select appropriate computational schemes and levels based on the actual researching objectives. MolSim-Transport offers three levels of schemes, 
![scale1](https://github.com/user-attachments/assets/c906babc-d6b6-42ac-87cf-1d5f51455c61)

![Scheme Table](https://github.com/user-attachments/assets/5b1cef53-877b-46ad-96b3-8da0e2143c06)

## **The QDHC Model Framework**
The QDHC model always adheres to a uniform structured workflow regardless of the precision level of simplification applied, which goes through an entire process from the initial geometric input final calculation of transport properties.

1. Problem Definition
2. Structural Input
3. Hamiltonian Matrix Construction
4. Definition of Electrode Interactions
5. Establishment of Device Green's Function
6. Calculation of Transport Properties
![flowchart](https://github.com/user-attachments/assets/0892594f-5379-45d1-b0dc-40871419ba35)

A detailed working protocol for the study of the transport property of 1,3-BDT molecule, known for its destructive quantum-interference feature, has been also posted to vividly demonstrated the proceeding process.
![workflow](https://github.com/user-attachments/assets/3e20bd3a-60d7-4947-b17d-b582c4dcf2b4)

## Benchmark studies
To validate the performance of _**MolSimTransport**_, it was applied to the following six molecular junction transport cases from the literature, forming a comprehensive benchmark test. These carefully selected cases span all levels of the QDHC strategy, ensuring the breadth and typicality of the test. The rapid replication of these cases demonstrates the high efficiency of the QDHC strategy in handling problems with different precision requirements and its unique ability to capture key factors in the transport process.

### System 1:  Controlled Quantum Interference in π-Stacked Dimers
- **Reference**: [Nature Chemistry, 2016, 8 (12), 1099–1104.](https://www.nature.com/articles/nchem.2588)
- **Model Treatment**: Bare molecule + EHMO.
![case1](https://github.com/user-attachments/assets/4f831fe5-934c-4dd3-a60a-8425baf48f60)

### System 2: Fano Resonance in Non-neutral molecule
- **Reference**: [Angewandte Chemie International Edition, 2022, 134 (40), e202210097.](https://onlinelibrary.wiley.com/doi/10.1002/anie.202210097)
- **Model Treatment**: Bare molecule + charge self-consistent EHMO.
![case2](https://github.com/user-attachments/assets/c33783d8-c968-47ea-a2f9-c1cd991f6442)

### System 3: Quantum Interference in Heterocyclic Molecule
- **Reference**: [Physical Chemistry Chemical Physics (PCCP), 2013, 16 (2), 653–662.](https://pubs.rsc.org/en/content/articlelanding/2014/cp/c3cp53866d)
- **Model Treatment**: Bare molecule + DFTB with higher accuracy.
![case3](https://github.com/user-attachments/assets/504db11e-d75c-4fb6-8baf-6b13f0dd359d)

### System 4: Binary conductance in the same junction for contact interface changes
- **Reference**: [Nature Nanotechnology, 2009, 4 (4), 230–234.](https://doi.org/10.1038/nnano.2009.10)
- **Model Treatment**: Extended molecule + DFTB with the same accuracy as the previous case.
![case4](https://github.com/user-attachments/assets/1a9f83d6-5767-4dd9-a895-d2645ffe2da3)

### System 5: Junctions with different govern conducting channels
- **Reference**: [Nano Letters, 2012, 12 (1), 354–358.](https://pubs.acs.org/doi/10.1021/nl203634m)
- **Model Treatment**: Device + energy level alignment to the Fermi level of the electrodes + DFTB with the same accuracy as the previous case.
![case5](https://github.com/user-attachments/assets/3a3cfc27-98e1-445f-a033-ca3dfaf03399)

### System 6: Biased Ferrocenyl Rectifier
- **Reference**: [Nature Communications, 2015, 6, 6324.](https://www.nature.com/articles/ncomms7324)
- **Model Treatment**: Device under external electric field + DFTB with the same accuracy as the previous case.
![case6](https://github.com/user-attachments/assets/86d6eb84-2aad-42b3-ace8-ef730191e340)


## **Repository Contents**
- The `MolSimTransport` directory contains the calculation scripts and atomic data files.

- The `share` directory contains the following:

  EM template files and cluster electrode structure files for the L2 scheme.

  Junction templates for different interfaces, pre-calculated Hamiltonian and surface Green's function files for the PL electrode, and scripts for calculating current and converting XYZ files to POSCAR files in the L3 scheme.

- The `test_file` directory holds test files for all three schemes, consistent with those provided in the manual.

- The `old-version-benchmark` directory contains benchmark test files from previous MATLAB versions(some are showcased in this README). You can also find these files in the `matlab-old-version` branch.
