#%%
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

cdim = 50
a = destroy(cdim).full()
adag = a.T.conj()
x = (a + adag)/np.sqrt(2)
p = -1j*(a-adag)/np.sqrt(2)
obs = x@x
u0 = coherent(cdim,0).full()

AL = np.linspace(-1,1,21)
def Dop(alpha, a):
    return expm(alpha*a.T.conj()-np.conjugate(alpha)*a)

data_id = np.zeros([len(AL),len(AL)], dtype = float)
for m in range(len(AL)):
    for n in range(len(AL)):
        th = AL[n] + 1j*AL[m]
        Dis = Dop(th, a)
        ud = Dis@u0
        data_id[m,n] = (ud.T.conj()@obs@ud)[0,0].real

X, Y = np.meshgrid(AL, AL)

# Create the plot
create_figure_cm(4,3.5)
plt.pcolormesh(X,Y,data_id, cmap="summer_r", shading='auto')
colorbar = plt.colorbar()
colorbar.ax.tick_params(labelsize=6)  
plt.tick_params(axis='both', which='major', labelsize=6)
plt.xticks([-1, 0, 1])  
plt.yticks([-1, 0, 1])  
print(f'max is {np.max(data_id)}')
print(f'min is {np.min(data_id)}')
plt.gca().set_aspect("equal")

# Add labels for better interpretation
# plt.xlabel(r'Re($\alpha$)')
# plt.ylabel(r'Im($\alpha$)')
# plt.savefig('fig2.pdf')
plt.show()

np.savez('x2_ideal.npz', obs=data_id)#save the ideal values

#%%
create_figure_cm(4,1.5)

plt.plot(AL,data_id[11,:],'k-', markersize=3)
plt.tick_params(axis='both', which='major', labelsize=6)
# plt.savefig('fig2.pdf')
plt.show()