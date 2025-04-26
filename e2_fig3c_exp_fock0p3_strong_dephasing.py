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

filepath = r"C:\Users\tanju\Dropbox\PY\Data\OREO\fig3 data\wigner0p3\01-46-25_Wigner1DPS.hdf5"


def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data_preselect = hdf["preselect"]
        data_preselect_cav = hdf["preselect_cav"]
        data_single_shot = hdf["single_shot"]
        cavity_drive_I = hdf["cavity_drive_I"]
        cavity_drive_Q = hdf["cavity_drive_Q"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data_preselect[:]),
            np.array(data_preselect_cav[:]),
            np.array(data_single_shot[:]),
            cavity_drive_I[:],
            cavity_drive_Q[:],
        )

(
    data_preselect_1_all,
    data_preselect_cav_all,
    data_single_shot_all,
    cavity_drive_I,
    cavity_drive_Q,
) = read_hdf5_file(filepath)
# print(data_preselect_1_all.shape, data_single_shot_all.shape)
# data_I.flatten().reshape(data_preselect_1_all.shape)
# print(data_I.shape)
# threshold = -1e-4

result_list = []
parity_matrices = []
obs_meas = []
print(data_preselect_1_all.shape)
print(np.average(data_preselect_1_all))
for phase in range(2):
    data_preselect_1 = data_preselect_1_all[:, :, :, phase]
    data_preselect_cav = data_preselect_cav_all[:, :, :, phase]
    data_single_shot = data_single_shot_all[:, :, :, phase]
    parity_matrix = []
    for c1 in range(data_preselect_1.shape[1]):
        p_is = []
        for c2 in range(data_preselect_1.shape[2]):
            data_single_shot_column = data_single_shot[:, c1, c2]
            data_preselect_1_column = data_preselect_1[:, c1, c2]
            data_preselect_cav_column = data_preselect_cav[:, c1, c2]
            # Preselect on ground state
            data_selected_0 = np.array(
                [
                    data_single_shot_column[i]
                    for i in range(len(data_single_shot_column))
                    if (
                        (data_preselect_1_column[i] == 0)
                        and (data_preselect_cav_column[i] == 0)
                    )
                ]
            )
            p = np.average(data_selected_0)  # "P(1|0)"
            p_is.append(p)
        parity_matrix.append(p_is)
    parity_matrices.append(parity_matrix)

parity_corrected = np.array(parity_matrices[0]) - np.array(parity_matrices[1])

parity_offset = -0.00018
parity_rescaling = 0.9506
parity_corrected = (
    np.array(parity_matrices[0]) - np.array(parity_matrices[1]) - parity_offset
) / parity_rescaling
print(f'max parity is {np.max(parity_corrected)}')
print(f'min parity is {np.min(parity_corrected)}')

#flipping
def flip_matrix_around_center(matrix):
    # Flip rows (reverse the entire matrix)
    matrix = matrix[::-1]
    # Flip columns (reverse each row)
    matrix = [row[::-1] for row in matrix]
    return np.array(matrix)
parity_corrected = flip_matrix_around_center(parity_corrected)

#plotting
x, y = np.meshgrid(cavity_drive_I, cavity_drive_Q)
# fig = plt.figure()
create_figure_cm(4,4)
# plt.xlabel(r"Re($\alpha$)")
# plt.ylabel(r"Im($\alpha$)")

#cut the edge
cavity_drive_I_trim = cavity_drive_I[1:-1]
cavity_drive_Q_trim = cavity_drive_Q[1:-1]
parity_corrected_trim = parity_corrected[1:-1,1:-1]

cf = plt.pcolormesh(
    cavity_drive_I_trim, cavity_drive_Q_trim, parity_corrected_trim, cmap="bwr", vmax=1, vmin=-1
)
# plt.title(f"Parity = {np.max(parity_corrected):.3f}")
plt.gca().set_aspect("equal")
plt.tick_params(axis='both', which='major', labelsize=6)
# plt.savefig('fig4_0p3.pdf')
plt.show()

#%%compute the map for the given displacement points, for state tomography
n_grid = len(cavity_drive_I_trim)
AL = np.zeros(n_grid**2, dtype=np.complex128)
cavity_drive_Q_trim_rev = cavity_drive_Q_trim[::-1]

w = 0
for i in range(n_grid):
    for k in range(n_grid):
        AL[w] = cavity_drive_I_trim[k] + 1j*cavity_drive_Q_trim_rev[i]
        w += 1

D = 6  # dimension to read
n_dis = n_grid**2 # no of displacement points

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
n_grid = len(cavity_drive_I_trim)

#the target state
# u = fock(cdim,0)
# u = fock(cdim,3)
# u = (fock(cdim,0) + fock(cdim,4)).unit()
u = (fock(cdim,0) + fock(cdim,3)).unit()
rho_tar = ket2dm(u)
# rho_tar = ket2dm(u).full()
# rho_tar[0,4] = 0.25
# rho_tar[4,0] = 0.25
# rho_tar = Qobj(rho_tar)

# rho_tar = thermal_dm(cdim, 0.242)

# Mapping
C = np.matmul(-W, beta[:, 0])
BETA = np.zeros([nD, n_grid**2 + 1])
BETA[:, 0] = C
BETA[:, 1 : n_grid**2 + 1] = W

# Target state function simulating GRAPE pulses and ideal displacements
def Y_target(rho_tar):
    rho_tar_D = Qobj(rho_tar[0:D, 0:D])  # no need 30 DIMS, CUT AT 6
    # Target observables
    Y_tar = np.zeros(nD)
    Y_tar[: D - 1] = np.diag(rho_tar_D.full()).real[:-1]  # Diagonal of rho
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
X_exp = np.zeros([1 + n_grid**2])
X_exp[0] = 1
X_exp[1:] = parity_corrected_trim.reshape(n_grid**2)

Y_tar, rho_tar_D = Y_target(rho_tar)

# Estimate the state by applying the inverse map to the experimental data
Y_est = np.zeros(nD)
Y_est = np.matmul(BETA, X_exp)
rho_est = rho_from_Y(Y_est)  # just a reshaping
Fmean, Fstd, qRho_est = Bysn_rho_v2(2**10, 1000*(nD), rho_tar_D.full(), rho_est.full())
F = fidelity(rho_tar_D, Qobj(qRho_est)) ** 2
print(f'Fidelity is {np.round(F, 2)}')
print(f'Fmean is {np.round(Fmean, 2)}')
print(f'Fstd is {np.round(Fstd, 2)}')

#%%nice Wigner plot
WW = wigner_parity_plot(qRho_est, 1.71, 51, [4,4])
np.savez('rho_f0p3_strong_dephasing.npz', rho=qRho_est)#save the ideal values

