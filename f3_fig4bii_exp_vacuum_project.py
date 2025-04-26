#%%first, get the Wigner experimental data
import os
# Specify the directory you want to change to
new_directory = r"C:\Users\tanju\Dropbox\PY\OREO"
os.chdir(new_directory)

from TK_basics import *

import numpy as np
import h5py
from qutip import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
start_time = time.time()#checking how long the code takes

filepath = r"C:\Users\tanju\Dropbox\PY\OREO\fig4 data\sparsewigner_vacuum_proj\14-22-38_StateProjectionSparseWigner2D.hdf5"


def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data_preselect = hdf["preselect_1"]
        data_preselect_cav = hdf["preselect_2"]
        data_preselect_I = hdf["I_p1"]
        data_preselect_cav_I = hdf["I_p2"]
        data_single_shot = hdf["single_shot"]
        disp_id = hdf["disp_id"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data_preselect[:]),
            np.array(data_preselect_cav[:]),
            np.array(data_preselect_I[:]),
            np.array(data_preselect_cav_I[:]),
            np.array(data_single_shot[:]),
            disp_id[:],
        )

(
    data_preselect_all,
    data_preselect_cav_all,
    data_preselect_I_all,
    data_preselect_cav_I_all,
    data_single_shot_all,
    disp_id,
) = read_hdf5_file(filepath)

parity_matrices = []
for phase in range(2):
    data_preselect_1 = data_preselect_all[:, :, phase]
    data_preselect_cav = data_preselect_cav_all[:, :, phase]
    data_single_shot = data_single_shot_all[:, :, phase]
    data_preselect_1_I = data_preselect_I_all[:, :, phase]
    data_preselect_cav_I = data_preselect_cav_I_all[:, :, phase]

    p_is = []
    for c1 in range(data_preselect_1.shape[1]):
        data_single_shot_column = data_single_shot[:, c1]
        data_preselect_1_column = data_preselect_1[:, c1]
        data_preselect_cav_column = data_preselect_cav[:, c1]
        data_preselect_1_column_I = data_preselect_1_I[:, c1]
        data_preselect_cav_column_I = data_preselect_cav_I[:, c1]
        # Preselect on ground state
        data_selected_0 = np.array(
            [
                data_single_shot_column[i]
                for i in range(len(data_single_shot_column))
                if (
                    (data_preselect_1_column_I[i] < -3e-4)
                    and (data_preselect_cav_column_I[i] < -2.8e-4)
                )
            ]
        )
        p = np.average(data_selected_0)  # "P(1|0)"
        # print(len(data_selected_0))
        p_is.append(p)
    parity_matrices.append(p_is)

parity_offset = -0.000700510792283095
parity_rescaling = 0.9474893575501806
parity_corrected = (
    np.array(parity_matrices[0]) - np.array(parity_matrices[1]) - parity_offset
) / parity_rescaling
print(f'max parity is {np.max(parity_corrected)}')
print(f'min parity is {np.min(parity_corrected)}')

#%%compute the map for the given displacement points (SPARSE), for state tomography
#the optimised displacement points
AL = np.array([-0.785123  +1.48657418j, -1.67098242+0.22666372j,
0.23477004+0.53155278j, -0.98121628-0.73382913j,
1.30034526-0.19658333j,  0.91057304-0.14736838j,
0.65327309-1.15538318j, -0.1952535 -0.0984287j ,
-0.36345514-1.25492752j, -0.86678666-0.27228837j,
0.50962262+0.81069781j,  0.69155442-0.69888758j,
-0.77472546+0.48025564j, -1.27414151-0.11396373j,
-0.04185742+0.91628513j,  0.25407875-1.70695007j,
0.00196135-0.56262436j,  0.00304715+0.18467356j,
0.19371779-0.92074751j, -1.2096629 -1.16284482j,
-1.159518  +0.56868829j,  0.59088723+0.14120895j,
0.84473373+0.45523965j,  0.49191403-0.32179364j,
-0.45441726-0.37142953j, -0.58965859+0.10145214j,
1.39193867-0.97823193j,  0.96166038+0.86471223j,
-0.24254956+0.54631778j,  1.67299726+0.31620016j,
-0.67203676+1.00506992j,  0.78077124+1.52604695j,
0.12879202-0.167258j  , -0.3002807 -0.8619771j ,
0.01810127+1.30174108j])

D = 6  # dimension to read
n_dis = D**2-1 # no of displacement points

nD = D**2 - 1  # no of parameters for general states
Ntr = D**2  # no of training for obtaining the map, at least D^2

cdim = 30  # truncation
a = destroy(cdim).full()  # annihilation for cavity
adag = a.T.conj()
P = expm(1j * np.pi * adag @ a)

# displacement operator
def Dis(alpha):
    Di = expm(alpha * adag - np.conj(alpha) * a)
    return Di

# this part if for obtaining the map
X_r = np.zeros([1 + n_dis, Ntr])  # store readouts
X_r[0, :] = np.ones([1, Ntr])  # setting the ones
Y_r = np.zeros([nD, Ntr])  # store the targets
for j in np.arange(0, Ntr):
    # qudit mixed state embedded in the cavity mode
    rd1 = np.zeros([cdim, cdim], dtype=np.complex128)
    u_rand = rand_ket(D)
    r_rand = (u_rand * u_rand.dag()).full()
    rd1[0:D, 0:D] = r_rand  # randRho(D)

    # assign targets
    cw = 0
    # diagonal elements
    for j1 in np.arange(0, D - 1):
        Y_r[cw, j] = rd1[j1, j1].real
        cw += 1
    # off-diagonal elements
    for j1 in np.arange(0, D - 1):
        for j2 in np.arange(j1 + 1, D):
            Y_r[cw, j] = rd1[j1, j2].real
            cw += 1
            Y_r[cw, j] = rd1[j1, j2].imag
            cw += 1

    w = 0
    for v in np.arange(0, n_dis):
        Di = Dis(AL[w])
        rt = Di.T.conj() @ rd1 @ Di
        X_r[w + 1, j] = np.trace(rt @ P).real 
        w += 1

# ridge regression
lamb = 0

# training, now to obtain the map
X_R = np.zeros([1 + nD, Ntr])  # will contain the parameters
X_R[0, :] = np.ones([1, Ntr])  # setting the ones
Y_R = np.zeros([n_dis, Ntr])  # will contain the obs

# re-defining variables
X_R[1 : nD + 1, :] = Y_r
Y_R[:, :] = X_r[1 : n_dis + 1, :]

Error, beta = QN_regression(X_R, Y_R, lamb)

M = beta[:, 1 : nD + 1]  # the map
W = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), np.transpose(M))
CN = np.linalg.norm(M, 2) * np.linalg.norm(W, 2)
print(f"Condition number is {CN}")

#%%state reconstruction with Bayesian inference 
cdim = 30 
D = 6#dimension to estimate
nD = D**2 - 1#no of state parameters

#the target state
# u = fock(cdim,0)
# u = fock(cdim,3)
# u = (fock(cdim,0) + fock(cdim,3)).unit()
u = (fock(cdim,0) + fock(cdim,4)).unit()
rho_tar = ket2dm(u)
# rho_tar = ket2dm(u).full()
# rho_tar[0,4] = 0.25
# rho_tar[4,0] = 0.25
# rho_tar = Qobj(rho_tar)

# rho_tar = thermal_dm(cdim, 0.242)

# Mapping
C = np.matmul(-W, beta[:, 0])
BETA = np.zeros([nD, nD + 1])
BETA[:, 0] = C
BETA[:, 1 : nD + 1] = W

# Target state function simulating GRAPE pulses and ideal displacements
def Y_target(rho_tar):
    rho_tar_D = Qobj(rho_tar[0:D, 0:D])  # no need 30 DIMS, CUT AT 6
    # Target observables
    Y_tar = np.zeros(nD)
    Y_tar[: D - 1] = np.diagonal(rho_tar_D.full()).real[:-1]  # Diagonal of rho
    off_diag = rho_tar_D[np.triu_indices(D, 1)]  # Upper triangle of rho
    Y_tar[D - 1 :: 2] = np.real(off_diag)
    Y_tar[D::2] = np.imag(off_diag)

    return Y_tar, rho_tar_D

# Builds a density matrix from the vector Y
def rho_from_Y(Y_est):
    rho_est = np.zeros([D, D], dtype=np.complex128)
    diagonal = np.append(Y_est[: D - 1], 1 - sum(Y_est[: D - 1]))
    np.fill_diagonal(rho_est, diagonal)  # Populate diagonal of rho

    index_i_list = np.triu_indices(D, 1)[0]
    index_j_list = np.triu_indices(D, 1)[1]
    for k in range(len(index_i_list)):  # Populate off-diagonals of rho
        index_i = index_i_list[k]
        index_j = index_j_list[k]
        rho_est[index_i, index_j] = Y_est[D - 1 + 2 * k] + 1j * Y_est[D + 2 * k]
        rho_est[index_j, index_i] = Y_est[D - 1 + 2 * k] - 1j * Y_est[D + 2 * k]

    return Qobj(rho_est)

# Experimental observables
X_exp = np.zeros([1 + nD])
X_exp[0] = 1
X_exp[1:] = parity_corrected#this is the parity for 35 points 

Y_tar, rho_tar_D = Y_target(rho_tar)

# Estimate the state by applying the inverse map to the experimental data
Y_est = np.zeros(nD)
Y_est = np.matmul(BETA, X_exp)
rho_est = rho_from_Y(Y_est)  # just a reshaping
Fmean, Fstd, qRho_est = Bysn_rho_v2(2**10, 1000*(nD), rho_tar_D.full(), rho_est.full())
F = fidelity(rho_tar_D, Qobj(qRho_est))**2
print(f'Fidelity is {np.round(F, 2)}')
print(f'Fmean is {np.round(Fmean, 2)}')
print(f'Fstd is {np.round(Fstd, 2)}')

#%%
WW = wigner_parity_plot(qRho_est, 1.71, 51, [4,4])
np.savez('rho_vacuum_proj_f0p4.npz', rho=qRho_est)#save the ideal values

