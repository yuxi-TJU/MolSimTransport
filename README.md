# **_MolSim-Transport_: A Question-Driven Hierarchical Computation Scheme for Efficiently Decoding Charge Transport in Molecular Junctions**

## **Introduction**
_**MolSim-Transport**_, a highly efficient computational scheme within the **Question-Driven Hierarchical Computation (QDHC)** framework, has been developed by [Professor Yu Xi's research group](http://39.107.98.173/) at Tianjin University. This scheme integrates several transport models across various levels of approximation, complemented by computational methods of different accuracies. It provides a comprehensive, multi-level solution for analyzing the charge transport properties in molecular junctions.

![hierarchy](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/e607529f-6e32-45fa-be8f-d2df5df58722)

The QDHC design in _**MolSim-Transport**_ bridges the theoretical gap between DFT+NEGF calculations on the full molecular devices and and simplified theoretical models, by encompassing a range of simplifications and approximations at various levels. Considering that the charge transport properties of molecular devices are jointly determined by the molecule, electrodes, and their interfacial interactions, the calculations accordingly span multiple physical scales from atoms to devices. Driven by the actual researching questions, this method selectively disregards minor factors being less impactful and only focuses on the primary ones that critically influence system behavior. Moreover, by effectively applying theoretical methods at different levels of approximation and precision, this strategy achieves greater efficiency while maintaining accuracy, thus enabling more effective exploration of charge transport behaviors in molecular junction systems.

## **Detailed Description of the QDHC Strategy**
The core of the QDHC (Question-Driven Hierarchical Computation) strategy lies in the refined stratification and approximation strategies at two critical levels: defining the computational system scope and selecting computational methods. Considering that a typical molecular device consists of three parts: the molecule, the molecule-electrode interface, and the source/drain electrodes, the QDHC strategy requires researchers to select appropriate computational schemes and levels based on the actual researching objectives. MolSim-Transport offers three levels of schemes, 

![Scheme Table](https://github.com/yuxi-TJU/MolSim-Transport/assets/68102657/0c693738-efcc-4a09-9723-256dff4b40b7)

## **The QDHC Model Framework**
The QDHC model always adheres to a uniform structured workflow regardless of the precision level of simplification applied, which goes through an entire process from the initial geometric input final calculation of transport properties.

1. Problem Definition
2. Structural Input
3. Hamiltonian Matrix Construction
4. Definition of Electrode Interactions
5. Establishment of Device Green's Function
6. Calculation of Transport Properties

![flowchart](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/bac5855f-ecce-4467-a208-5a6b774f8ec0)

A detailed working protocol for the study of the transport property of 1,3-BDT molecule, known for its destructive quantum-interference feature, has been also posted to vividly demonstrated the proceeding process.
![workflow](https://github.com/yuxi-TJU/Question-oriented-Hierarchical-Transport-Model/assets/68102657/ad6b807a-0ff2-4671-bfeb-35191da5049a)


## Benchmark studies
To validate the performance of _**MolSim-Transport**_, it was applied to the following six molecular junction transport cases from the literature, forming a comprehensive benchmark test. These carefully selected cases span all levels of the QDHC strategy, ensuring the breadth and typicality of the test. The rapid replication of these cases demonstrates the high efficiency of the QDHC strategy in handling problems with different precision requirements and its unique ability to capture key factors in the transport process.

### System 1:  Controlled Quantum Interference in π-Stacked Dimers
- **Reference**: [Nature Chemistry, 2016, 8 (12), 1099–1104.](https://www.nature.com/articles/nchem.2615)
- **Model Treatment**: Bare molecule + EHMO.
![case1](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/06c971bf-71f8-4d0e-aeba-772bd2884e5d)

### System 2: Fano Resonance in Non-neutral molecule
- **Reference**: [Angewandte Chemie International Edition, 2022, 134 (40), e202210097.](https://onlinelibrary.wiley.com/doi/10.1002/anie.202210097)
- **Model Treatment**: Bare molecule + charge self-consistent EHMO.
![case2](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/2d3e9b53-9f78-499a-831d-7fc74e4ba4ba)

### System 3: Quantum Interference in Heterocyclic Molecule
- **Reference**: [Physical Chemistry Chemical Physics (PCCP), 2013, 16 (2), 653–662.](https://pubs.rsc.org/en/content/articlelanding/2014/cp/c3cp53866d)
- **Model Treatment**: Bare molecule + DFTB with higher accuracy.
![case3](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/8f20778c-b85c-4514-abdb-afed86db17c1)

### System 4: Binary conductance in the same junction for contact interface changes
- **Reference**: [Nature Nanotechnology, 2009, 4 (4), 230–234.](https://doi.org/10.1038/nnano.2009.10)
- **Model Treatment**: Extended molecule + DFTB with the same accuracy as the previous case.
![case4](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/c2c1eb2f-46c1-4892-9290-415acbf4b7ec)

### System 5: Junctions with different govern conducting channels
- **Reference**: [Nano Letters, 2012, 12 (1), 354–358.](https://pubs.acs.org/doi/10.1021/nl203634m)
- **Model Treatment**: Device + energy level alignment to the Fermi level of the electrodes + DFTB with the same accuracy as the previous case.
![case5](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/08035492-a13e-4c8d-aeea-3058e56529eb)

### System 6: Biased Ferrocenyl Rectifier
- **Reference**: [Nature Communications, 2015, 6, 6324.](https://www.nature.com/articles/ncomms7324)
- **Model Treatment**: Device under external electric field + DFTB with the same accuracy as the previous case.
![case6](https://github.com/yuxi-TJU/Question-driven-Hierarchical-Transport-Model/assets/68102657/199efa73-40d8-45e9-869c-2d5a6dead373)
