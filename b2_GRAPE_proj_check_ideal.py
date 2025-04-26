#%%
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import scipy as sc
from scipy.linalg import expm
import os
new_directory = r"C:\Users\tanju\Dropbox\PY\OREO"
os.chdir(new_directory)
from TK_basics import *

# Simulation Dimensions
cdim = 20
qdim = 2
ug = basis(qdim,0).full()
ue = basis(qdim,1).full()

ind = 0
data_dr = np.load(rf"C:\Users\tanju\Dropbox\PY\OREO\projbinomial04D10\{ind}\waves.npz", "r")

dt = data_dr['dt']
# the drives are already in GHz
qubitI = data_dr['QubitI']
qubitQ = data_dr['QubitQ']
cavI = data_dr['CavityI']
cavQ = data_dr['CavityQ']

# Mode Operators
q = destroy(qdim).full()
c = destroy(cdim).full()
qd, cd = q.T.conj(), c.T.conj()

Q = np.kron(q, np.eye(cdim))
C = np.kron(np.eye(qdim), c)
Qd, Cd = Q.T.conj(), C.T.conj()

# Hamiltonian Parameters in GHz
chi = 1.482e-3
Kerr = 4.4e-6
chi_prime = 13.6e-6
anharm = 0*175.31e-3

# Drift Hamiltonian
H0 = -2*np.pi*chi*Cd@C@Qd@Q - 2*np.pi*Kerr/2*Cd@Cd@C@C - 2*np.pi*chi_prime/2*Cd@Cd@C@C@Qd@Q - 2*np.pi*anharm/2*Qd@Qd@Q@Q

Hc = [
        2*np.pi*(C + Cd),
        1j*2*np.pi*(C - Cd),
        2*np.pi*(Q + Qd),
        1j*2*np.pi*(Q - Qd),
        ]

Utot = np.kron(np.eye(qdim),np.eye(cdim))
for j in range(len(qubitI)):
    Udt = expm(-1j*(H0+qubitI[j]*Hc[2]+qubitQ[j]*Hc[3]+cavI[j]*Hc[0]+cavQ[j]*Hc[1]))
    Utot = Udt@Utot

D = 10
u = (fock(cdim,0)+fock(cdim,4)).unit()
G_tar = ket2dm(u)[0:D,0:D]

Utotcav = Utot[0:cdim, 0:cdim]#<g|U_m|g>
Gtot = Utotcav.T.conj()@Utotcav
G = Gtot[0:D,0:D]#the truncated U from simulation

cost = np.sum(np.abs(G_tar-G)**2)/D**2
print(f'cost is {cost}')

np.savez('Matrix_Um.npz', Um=Utot)#save the matrix
#%%checking observables for random states within D
x_op = np.kron(np.eye(qdim),ket2dm(u).full())

Ntr = 200
xe_id = np.zeros(Ntr, dtype = float)
xe_sim = np.zeros(Ntr, dtype = float)
for j in range(Ntr):
    #qudit state embedded in the cavity mode
    rd1 = np.zeros([cdim, cdim], dtype = np.complex128)
    u_rand = rand_ket(D)
    r_rand = (u_rand*u_rand.dag()).full()
    rd1[0:D,0:D] = r_rand
    
    rho = np.kron(ug@ug.T.conj(), rd1)
    xe_id[j] = np.real(np.trace(x_op@rho))
    #the evolution
    rhot = Utot@rho@Utot.T.conj()
    xe_sim[j] = np.real( np.trace(np.kron(ug@ug.T.conj(),np.eye(cdim))@rhot) )

plt.plot(xe_id, xe_sim,'ok')
plt.plot(xe_id, xe_id,'r-')
plt.xlabel('x_id')
plt.ylabel('x_sim')

plt.show()
