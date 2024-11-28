# Case 1 - L1 scheme - molecule+EHT
> In the study published in _Nature Chemistry_ 2016 titled "Mechanically controlled quantum interference in individual π-stacked dimers, van der Zant, et al. have successfully achieved quasi-periodic regulation of quantum interference effects in π-stacked dimers by controlling the relative motion of metal electrodes using mechanically controlled broken junction (MCBJ) technology.

Since this research focuses on the intrinsic property of the molecule and effect of conformational changes, the simpler computational scheme **L1** was used. 

The research system was simplified to a molecular dimer, without explicitly considering the electrodes (using the wide-band approximation and **constant self-energy** instead). Two hundred conformations were constructed, covering the continuous stretching process from complete overlap to staggered conformations. The **Extended Hückel Theory (EHT)** method was used to describe the electronic structure of these conformations, and their transmission spectra were superimposed to form a two-dimensional transmission spectrum that varied with stretching distance. 

This image successfully replicated the periodic conductance oscillations reported in the original publication, which are caused by quantum interference effects between molecules as the electrodes are stretched. Calculating one of the transmission spectra took only 10 seconds on a personal computer, and reproducing the entire two-dimensional image took about 30 minutes (including substantial data transfer time).

![case1](https://github.com/yuxi-TJU/MolSim-Transport/assets/68102657/b76ad5aa-b41b-4670-8218-22fc46a081af)
