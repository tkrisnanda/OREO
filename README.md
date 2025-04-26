This repository contains codes used in the paper, "Direct estimation of arbitrary observables of an oscillator" (https://arxiv.org/abs/2503.10436).
For experimental data, download from (https://www.dropbox.com/scl/fo/v218w0md799kwbkwbs385/AJzwhegKpLEOirU1c1USPYs?rlkey=rbn56b0uwofad4k823jsng7ti&st=4iz2y0aw&dl=0).

The main python codes are listed as a1_...py, a2_...py, ...

a1_...py generates the experimental pulses required to implement the mapping unitary Um for general observable
a2_...py tests the pulses generated in a1_...py for ideal cases without imperfections
b1_...py generates the experimental pulses required to implement the mapping unitary Um for positive observable (projection operator)
b2_...py tests the pulses generated in b1_...py for ideal cases without imperfections
c1_...py generates the experimental pulses required to implement the reverse unitary Ur given Um generated from b1_...py
c2_...py tests the pulses generated in c1_...py for ideal cases without imperfections, as well as the whole sequence of the projection protocol
