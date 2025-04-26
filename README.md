This repository contains codes used in the paper, "Direct estimation of arbitrary observables of an oscillator" (https://arxiv.org/abs/2503.10436).
For experimental data, download from (https://www.dropbox.com/scl/fo/v218w0md799kwbkwbs385/AJzwhegKpLEOirU1c1USPYs?rlkey=rbn56b0uwofad4k823jsng7ti&st=4iz2y0aw&dl=0).
 
The main python codes are listed as a1_...py, a2_...py, ...

For generating pulses, we modified the original GRAPE package developed at Yale. Here, the modified package is in the 'grape' folder.
Before running the codes, please read 'How to use OREO in you experiment' in the Supplemental Material of our paper.
Here are the codes for generating pulses and testing them:
1. a1_...py generates the experimental pulses required to implement the mapping unitary Um for general observable
2. a2_...py tests the pulses generated in a1_...py for ideal cases without imperfections
3. b1_...py generates the experimental pulses required to implement the mapping unitary Um for positive observable (e.g., projection operator)
4. b2_...py tests the pulses generated in b1_...py for ideal cases without imperfections
5. c1_...py generates the experimental pulses required to implement the reverse unitary Ur given Um generated from b1_...py
6. c2_...py tests the pulses generated in c1_...py for ideal cases without imperfections, as well as the whole sequence of the projection protocol

Here are the codes for processing the experimental data and plotting the figures in our paper:
7. d1_...py computes ideal values for <x> 
8. d2_...py processes exp data and plots <x> in Fig. 2 
9. d3_...py computes ideal values for <x^2> 
10. d4_...py processes exp data and plots <x^2> in Fig. 2 
11. d5_...py computes ideal values for <x^2+p> 
12. d6_...py processes exp data and plots <x^2+p> in Fig. 2 
13. d7_...py computes ideal values for <(x^2+p)^2> 
14. d8_...py processes exp data and plots <(x^2+p)^2> in Fig. 2 

15. e1_...py processes the Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for Fock 3 in Fig. 3a 
16. e2_...py processes the Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for Fock 0p3 (with strong dephasing) in Fig. 3c 
17. e3_...py processes the sparse Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for Fock 0p3 (with weak dephasing) 
18. e4_...py processes the observable data for Fock 3 in Fig. 3b 
19. e5_...py processes the observable data for Fock 0p3 (with strong dephasing) in Fig. 3b 
20. e6_...py processes the observable data for Fock 0p3 (with weak dephasing) in Fig. 3b 

21. f1_...py processes the sparse Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for vacuum in Fig. 4bi
22. f2_...py processes the sparse Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for thermal state in Fig. 4ci
23. f3_...py processes the sparse Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for vacuum projected into binomial state 0p4 in Fig. 4bii
24. f4_...py processes the sparse Wigner tomography data, produces estimated state via Bayesian inference, plot the reconstructed Wigner for thermal state projected into binomial state 0p4 in Fig. 4cii
