#%%
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

#scaling from qubit singleshot measurement (pe)
data_floor = 0.02
data_scaling = 0.96-data_floor
def scale_q(data):
    return (data-data_floor)/data_scaling

#factor
D = 10
a = destroy(15).full()
x = (a+a.T.conj())/np.sqrt(2)
p = -1j*(a-a.T.conj())/np.sqrt(2)
obs = x
obs = obs[0:D,0:D]
f = (np.max(np.linalg.eigvals(obs))).real
print(f'f is {f}')

filepath = r"C:\Users\tanju\Dropbox\PY\OREO\fig2 data\x\13-36-54_ObservableMeasPS.hdf5"

file = h5py.File(filepath, "r")
data = file["single_shot"]
data_size = data.shape
print(f'data size {data_size}')
data_ave = np.mean(data, axis=0)#this is pe
data_ave = (1-2*scale_q(data_ave))*f

AL = np.linspace(-1,1,21)
X, Y = np.meshgrid(-AL, -AL)#the minus sign is because our displacement is reversed

create_figure_cm(4.3,3.5)
plt.pcolormesh(X,Y,data_ave, cmap="summer_r", shading='auto')#, extent=[AL.min(), AL.max(), AL.min(), AL.max()], origin='lower')

plt.tick_params(axis='both', which='major', labelsize=6)
plt.xticks([-1, 0, 1])  
plt.yticks([-1, 0, 1])  
plt.clim(-np.sqrt(2), np.sqrt(2))
print(f'max is {np.max(data_ave)}')
print(f'min is {np.min(data_ave)}')

colorbar = plt.colorbar()
colorbar.ax.tick_params(labelsize=6)  
colorbar.set_ticks([-np.sqrt(2), 0, np.sqrt(2)])
colorbar.set_ticklabels(['1.0', '1.0', '1.0'])

plt.gca().set_aspect("equal")

# Add labels for better interpretation
# plt.xlabel(r'Re($\alpha$)')
# plt.ylabel(r'Im($\alpha$)')
# plt.savefig('fig2a_up.pdf')
plt.show()
#%%1D plot with bootstrap
ind = 10
def Bootstrap(ind):
    data_1d = data[:,20-ind,:]
    le = data_1d.shape[1]

    NBS = 20

    data_BS_all = np.zeros([NBS, le])
    for j in range(NBS):
        for k in range(le):
            data_BS = np.random.choice(data_1d[:,k], size=1000, replace=True)
            data_BS_all[j,k] = np.mean(data_BS)#pe

    data_BS_all = (1-2*scale_q(data_BS_all))*f

    data_BS_ave = np.mean(data_BS_all, axis=0)
    data_BS_std = np.std(data_BS_all, axis=0)
    return data_BS_ave, data_BS_std

data_BS_ave1, data_BS_std1 = Bootstrap(10)#mid
data_BS_ave2, data_BS_std2 = Bootstrap(20)#top

create_figure_cm(3.45,1.5)

#recall the ideal data
data_id = np.load(r'C:\Users\tanju\Dropbox\PY\OREO\x_ideal.npz')['obs']

plt.errorbar(-AL[::2], data_BS_ave1[::2], data_BS_std1[::2], fmt='o', color='#50c8ef', capsize=0, markersize=2, alpha = 1, elinewidth=1)
plt.errorbar(-AL[::2], data_BS_ave2[::2], data_BS_std2[::2], fmt='o', color='#ed2c85', capsize=0, markersize=2, alpha = 1, elinewidth=1)
plt.plot(AL, data_id[ind,:],'k-', linewidth=1, zorder=1)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.xlim([-1.05,1.05])
plt.yticks([-np.sqrt(2), 0, np.sqrt(2)], ['1','1','1'])
# plt.savefig('fig2a_bot.pdf')
plt.show()



