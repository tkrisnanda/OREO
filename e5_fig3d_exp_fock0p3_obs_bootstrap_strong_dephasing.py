#%%
import os
# Specify the directory you want to change to
new_directory = r"C:\Users\tanju\Dropbox\PY\OREO"
os.chdir(new_directory)

from TK_basics import *

import numpy as np
from qutip import *
from scipy.linalg import expm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
start_time = time.time()#checking how long the code takes

import h5py

rho_cav = np.load(r"C:\Users\tanju\Dropbox\PY\OREO\rho_f0p3_strong_dephasing.npz")['rho']
#the non-Gaussianity threshold
data = np.array([[-1.5       ,  0.77325729],
       [-1.45      ,  0.77387468],
       [-1.4       ,  0.77451667],
       [-1.35      ,  0.7751847 ],
       [-1.3       ,  0.77588037],
       [-1.25      ,  0.77660539],
       [-1.2       ,  0.7773616 ],
       [-1.15      ,  0.77815101],
       [-1.1       ,  0.77897578],
       [-1.05      ,  0.77983827],
       [-1.        ,  0.78074104],
       [-0.95      ,  0.78168688],
       [-0.9       ,  0.78267883],
       [-0.85      ,  0.7837202 ],
       [-0.8       ,  0.78481462],
       [-0.75      ,  0.78596605],
       [-0.7       ,  0.78717884],
       [-0.65      ,  0.78845774],
       [-0.6       ,  0.78980798],
       [-0.55      ,  0.79123532],
       [-0.5       ,  0.79274607],
       [-0.45      ,  0.79434721],
       [-0.4       ,  0.79604638],
       [-0.35      ,  0.79785206],
       [-0.3       ,  0.79977356],
       [-0.25      ,  0.80182115],
       [-0.2       ,  0.80400613],
       [-0.15      ,  0.80634096],
       [-0.1       ,  0.8088393 ],
       [-0.05      ,  0.81151616],
       [ 0.        ,  0.81438792],
       [ 0.05      ,  0.81747243],
       [ 0.1       ,  0.82078907],
       [ 0.15      ,  0.82435872],
       [ 0.2       ,  0.82820374],
       [ 0.25      ,  0.83234794],
       [ 0.3       ,  0.8368164 ],
       [ 0.35      ,  0.84163528],
       [ 0.4       ,  0.84683163],
       [ 0.45      ,  0.85243301],
       [ 0.5       ,  0.85846719],
       [ 0.55      ,  0.86496175],
       [ 0.6       ,  0.87194367],
       [ 0.65      ,  0.87943893],
       [ 0.7       ,  0.88747204],
       [ 0.75      ,  0.89606565],
       [ 0.8       ,  0.90524005],
       [ 0.85      ,  0.91501276],
       [ 0.9       ,  0.925398  ],
       [ 0.95      ,  0.93640626],
       [ 1.        ,  0.94804387],
       [ 1.05      ,  0.96031262],
       [ 1.1       ,  0.97320959],
       [ 1.15      ,  0.98672708],
       [ 1.2       ,  1.0008528 ],
       [ 1.25      ,  1.0155702 ],
       [ 1.3       ,  1.03085911],
       [ 1.35      ,  1.04669632],
       [ 1.4       ,  1.06305644],
       [ 1.45      ,  1.07991255],
       [ 1.5       ,  1.09723701]])
la = data[:,0]
F = data[:,1]

cdim = 30
cavT1 = 1e6 #in ns
# tau = np.array([1e-3,10,50,100,200])*1e3#the wait time in ns
tau = np.array([1e-3])*1e3#the wait time in ns
c = destroy(cdim)
cd = c.dag()
# Collapse Operators
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
        P[j, k] = (rho_f[0,3] + rho_f[3,0] + la[k]*rho_f[4,4]).real

#%%now the experimental data using OREO
# Model function
def cos_model(theta, A, B, C, D):
    return A * np.cos(theta*B + C) + D #+ B
initial_guess = [1.0, 2*np.pi*3, 0, 0]  # Initial guess for A, C, B

NBS = 20#number of bootstrapping

#scaling from qubit singleshot measurement (pe)
data_floor = 0.02
data_scaling = 0.96-data_floor
def scale_q(data):
    return (data-data_floor)/data_scaling

LAMBDAS = [-1, -0.5, 0, 0.5, 1]

filepath = r"C:\Users\tanju\Dropbox\PY\Data\OREO\fig3 data\fock0p3_obs_strong_dephasing\17-14-44_ObservableMeasNonGauss.hdf5"

def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as hdf:
        data_preselect = hdf["preselect"]
        data_preselect_cav = hdf["preselect_cav"]
        data_single_shot = hdf["single_shot"]
        lambda_id = hdf["lambda_id"]
        pulse_rot = hdf["pulse_rot"]
        # print([x[0] for x in data[:]])
        return (
            np.array(data_preselect[:]),
            np.array(data_preselect_cav[:]),
            np.array(data_single_shot[:]),
            lambda_id[:],
            pulse_rot[:],
        )

(
    data_preselect_all,
    data_preselect_cav_all,
    data_single_shot_all,
    lambda_id,
    pulse_rot,
) = read_hdf5_file(filepath)

obs_BS_all = np.zeros([NBS, len(lambda_id), len(pulse_rot)], dtype=float)
amp = np.zeros([NBS, len(lambda_id)], dtype=float)
for j in range(NBS):
    obs_meas = []
    for lamb in range(len(lambda_id)):
        Opr = np.zeros(len(pulse_rot), dtype=float)
        for t in range(len(pulse_rot)):
            data_preselect = data_preselect_all[:, lamb, t]
            data_preselect_cav = data_preselect_cav_all[:, lamb, t]
            data_single_shot = data_single_shot_all[:, lamb, t]

            # print(np.average(data_preselect))

            obs_selected = np.array(
                [
                    data_single_shot[i]
                    for i in range(len(data_preselect))
                    if data_preselect[i] == 0 and data_preselect_cav[i] == 0
                ]
            )
            # print(len(obs_selected))
            # obs_selected_BS = obs_selected
            obs_selected_BS = np.random.choice(obs_selected, size=len(obs_selected), replace=True)
            obs_meas.append(np.average(obs_selected_BS))

            Opr[t] = 1-2*scale_q(np.average(obs_selected_BS))
        params, covariance = curve_fit(cos_model, pulse_rot, Opr, p0=initial_guess)
        amp[j, lamb] = np.abs(params[0])

    obs_meas = np.array(obs_meas)
    # obs = 1 - 2 * np.array(obs_meas)
    obs = 1 - 2 * np.array(scale_q(obs_meas))
    obs_BS_all[j,:,:] = obs.reshape(len(lambda_id), len(pulse_rot))

amp_mean = np.mean(amp, axis=0)
amp_std = np.std(amp, axis=0)

create_figure_cm(4.5,4)

plt.plot(la, F,'k', linewidth=1)
plt.plot(la, P[0,:],':', color='#4dbd98', linewidth=1, alpha = 1)
plt.errorbar(LAMBDAS,amp_mean, amp_std, fmt='^', color='#4dbd98', capsize=0, markersize=2, alpha = 1, elinewidth=1)
plt.xlim([-1.05,1.05])
plt.ylim([0.2,1])
plt.tick_params(axis='both', which='major', labelsize=6)
# plt.savefig('fig4_0p3_obs_bad_cosfit.pdf')
plt.show()



# %%
