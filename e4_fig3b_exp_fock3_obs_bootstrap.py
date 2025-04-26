#%%simulation from full-state tomography
import os
# Specify the directory you want to change to
new_directory = r"C:\Users\tanju\Dropbox\PY\OREO"
os.chdir(new_directory)

from TK_basics import *

import numpy as np
from qutip import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
start_time = time.time()#checking how long the code takes

import h5py

rho_cav = np.load(r"C:\Users\tanju\Dropbox\PY\OREO\rho_f3.npz")['rho']
#the non-Gaussianity threshold
data = np.array([[-1.5  ,    0.51632097],
 [-1.45 ,    0.5172345 ],
 [-1.4  ,    0.51819191],
 [-1.35  ,   0.51919633],
 [-1.3  ,    0.52025121],
 [-1.25  ,   0.5213603 ],
 [-1.2  ,    0.52252772],
 [-1.15  ,   0.52375802],
 [-1.1   ,   0.52505619],
 [-1.05  ,   0.52642774],
 [-1.    ,   0.52787874],
 [-0.95  ,   0.52941594],
 [-0.9    ,  0.53104678],
 [-0.85  ,   0.53277955],
 [-0.8  ,    0.53462344],
 [-0.75,     0.5365887 ],
 [-0.7   ,   0.53868671],
 [-0.65  ,   0.54093018],
 [-0.6   ,   0.54333331],
 [-0.55  ,   0.54591189],
 [-0.5   ,   0.54868359],
 [-0.45  ,   0.55166806],
 [-0.4    ,  0.5548872 ],
 [-0.35  ,   0.5583653 ],
 [-0.3   ,   0.56212924],
 [-0.25  ,   0.5662086 ],
 [-0.2   ,   0.57063572],
 [-0.15  ,   0.57544569],
 [-0.1   ,   0.58067608],
 [-0.05  ,   0.58636666],
 [ 0.     ,  0.5925587 ],
 [ 0.05  ,   0.59929417],
 [ 0.1   ,   0.60661461],
 [ 0.15  ,   0.61455973],
 [ 0.2    ,  0.623166  ],
 [ 0.25   ,  0.63246509],
 [ 0.3   ,   0.64248251],
 [ 0.35  ,   0.65323652],
 [ 0.4   ,   0.66473737],
 [ 0.45   ,  0.67698712],
 [ 0.5   ,   0.68997977],
 [ 0.55  ,   0.70370187],
 [ 0.6   ,   0.7181334 ],
 [ 0.65  ,   0.73324883],
 [ 0.7   ,   0.74901829],
 [ 0.75  ,   0.76540874],
 [ 0.8   ,   0.78238508],
 [ 0.85  ,   0.79991115],
 [ 0.9   ,   0.8179506 ],
 [ 0.95  ,   0.8364676 ],
 [ 1.    ,   0.85542741],
 [ 1.05  ,   0.87479677],
 [ 1.1  ,    0.89454423],
 [ 1.15  ,   0.91464026],
 [ 1.2   ,   0.93505736],
 [ 1.25  ,   0.95577005],
 [ 1.3   ,   0.97675488],
 [ 1.35   ,  0.99799026],
 [ 1.4    ,  1.01945644],
 [ 1.45   ,  1.04113535],
 [ 1.5    ,  1.06301048]])

la = data[:,0]
F = data[:,1]

cdim = 30
cavT1 = 1e6 #in ns
# tau = np.array([1e-3,10,50,100,200])*1e3#the wait time in ns
tau = np.array([1e-3])*1e3#the wait time in ns
c = destroy(cdim)
cd = c.dag()
# Collapse Operators for cavity only
nbar_cav = 0
c_ops = [
    np.sqrt((1 + nbar_cav) / cavT1) * c,  # Cavity Relaxation
    np.sqrt(nbar_cav / cavT1) * cd,  # Cavity Thermal Excitations
]
H = qeye(cdim)
initial = np.zeros([cdim, cdim], dtype = np.complex128)
D = rho_cav.shape[0]
initial[0:D, 0:D] = rho_cav#reconstructed state
initial = Qobj(initial)

P = np.zeros([len(tau),len(la)], dtype=float)
for j in range(len(tau)):
    tlist = np.linspace(0,tau[j],21)
    # Dynamics
    results = mesolve(
        H,
        initial,
        tlist,
        c_ops=c_ops,
    )
    rho_f = results.states[-1]
    for k in range(len(la)):
        P[j, k] = (rho_f[3,3] + la[k]*rho_f[4,4]).real

#%%now the experimental data using OREO

NBS = 20#number of bootstrapping

#scaling from qubit singleshot measurement (pe)
data_floor = 0.02
data_scaling = 0.96-data_floor
def scale_q(dataa):
    return (dataa-data_floor)/data_scaling

LAMBDAS = [-1, -0.5, 0, 0.5, 1]

filepath_one = r"C:\Users\tanju\Dropbox\PY\Data\OREO\fig3 data\fock3_obs\14-30-28_ObservableMeasNonGauss.hdf5"
filepath = r"C:\Users\tanju\Dropbox\PY\Data\OREO\fig3 data\fock3_obs\14-36-12_ObservableMeasNonGauss.hdf5"

def read_hdf5_file_one(file_path):
    with h5py.File(file_path, "r") as hdf:
        data_preselect = hdf["preselect"]
        data_preselect_cav = hdf["preselect_cav"]
        data_single_shot = hdf["single_shot"]
        lambda_id = hdf["lambda_id"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data_preselect[:]),
            np.array(data_preselect_cav[:]),
            np.array(data_single_shot[:]),
            lambda_id[:],
        )
def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data_preselect = hdf["preselect"]
        data_preselect_cav = hdf["preselect_cav"]
        data_single_shot = hdf["single_shot"]
        lambda_id = hdf["lambda_id"]
        decay_time = hdf["decay_time"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data_preselect[:]),
            np.array(data_preselect_cav[:]),
            np.array(data_single_shot[:]),
            lambda_id[:],
            np.array(decay_time[:]),
        )

(
    data_preselect_all_one,
    data_preselect_cav_all_one,
    data_single_shot_all_one,
    lambda_id_one,
) = read_hdf5_file_one(filepath_one)
(
    data_preselect_all,
    data_preselect_cav_all,
    data_single_shot_all,
    lambda_id,
    decay_time,
) = read_hdf5_file(filepath)

# print(data_preselect_all.shape)

obs_BS_all_one = np.zeros([NBS, len(lambda_id_one)], dtype=float)
for j in range(NBS):
    obs_meas_one = []
    for lamb in range(data_single_shot_all_one.shape[1]):
        data_preselect = data_preselect_all_one[:, lamb]
        data_preselect_cav = data_preselect_cav_all_one[:, lamb]
        data_single_shot = data_single_shot_all_one[:, lamb]

        obs_selected = np.array(
            [
                data_single_shot[i]
                for i in range(len(data_preselect))
                if data_preselect[i] == 0 and data_preselect_cav[i] == 0
            ]
        )
        obs_selected_BS = np.random.choice(obs_selected, size=1000, replace=True)
        obs_meas_one.append(np.average(obs_selected_BS))

    obs_meas_one = np.array(obs_meas_one)
    # obs = 1 - 2 * np.array(obs_meas)
    obs_one = 1 - 2 * np.array(scale_q(obs_meas_one))
    obs_BS_all_one[j,:] = obs_one

obs_BS_all_mean_one = np.mean(obs_BS_all_one, axis=0)
obs_BS_all_std_one = np.std(obs_BS_all_one, axis=0)

obs_BS_all = np.zeros([NBS, len(decay_time), len(lambda_id)], dtype=float)
for j in range(NBS):
    obs_meas = []
    for t in range(len(decay_time)):
        for lamb in range(len(lambda_id)):
            data_preselect = data_preselect_all[:, t, lamb]
            data_preselect_cav = data_preselect_cav_all[:, t, lamb]
            data_single_shot = data_single_shot_all[:, t, lamb]

            # print(np.average(data_preselect))

            obs_selected = np.array(
                [
                    data_single_shot[i]
                    for i in range(len(data_preselect))
                    if data_preselect[i] == 0 and data_preselect_cav[i] == 0
                ]
            )
            obs_selected_BS = np.random.choice(obs_selected, size=1000, replace=True)
            obs_meas.append(np.average(obs_selected_BS))
    obs_meas = np.array(obs_meas)
    # obs = 1 - 2 * np.array(obs_meas)
    obs = 1 - 2 * np.array(scale_q(obs_meas))
    obs_BS_all[j,:,:] = obs.reshape(len(decay_time), len(lambda_id))

obs_BS_all_mean = np.mean(obs_BS_all, axis=0)
obs_BS_all_std = np.std(obs_BS_all, axis=0)

create_figure_cm(4.5,4)

plt.plot(la, F,'k', linewidth=1)
plt.plot(la, P[0,:],':', color='#26547c', linewidth=1)
# plt.plot(la, P[1,:],'g:')
# plt.plot(la, P[2,:],'c:')
# plt.plot(la, P[3,:],'m:')
# plt.plot(la, P[4,:],'r:')
# plt.plot(la, P[5,:],'r:')

plt.errorbar(
        LAMBDAS,
        obs_BS_all_mean_one, obs_BS_all_std_one, fmt='o', color='#26557d', capsize=0, markersize=2, alpha = 1, elinewidth=1
    )
# plt.errorbar(
#     LAMBDAS,
#     obs_BS_all_mean[0, :], obs_BS_all_std[0, :], fmt='o', c='g', capsize=3, markersize=3
# )
plt.errorbar(
    LAMBDAS,
    obs_BS_all_mean[1, :], obs_BS_all_std[1, :], fmt='s', color='#ef4870', capsize=0, markersize=2, alpha = 0.8, elinewidth=1
)
plt.errorbar(
    LAMBDAS,
    obs_BS_all_mean[2, :], obs_BS_all_std[2, :], fmt='v', color='#ffd168', capsize=0, markersize=2, alpha = 0.6, elinewidth=1
)
plt.errorbar(
    LAMBDAS,
    obs_BS_all_mean[3, :], obs_BS_all_std[3, :], fmt='^', color='#4dbd98', capsize=0, markersize=2, alpha = 0.4, elinewidth=1
)

plt.xlim([-1.05,1.05])
plt.ylim([0.2,1])
# plt.xlabel(r'$\lambda$', fontsize=6)
# plt.ylabel(r'$p_{3} + \lambda p_{4}$', fontsize=6)
plt.tick_params(axis='both', which='major', labelsize=6)
# plt.savefig('fig4_fock3_obs.pdf')
plt.show()


# %%
