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
data_dr_rev = np.load(rf"C:\Users\tanju\Dropbox\PY\OREO\projbinomial04D6_rev\{ind}\waves.npz", "r")

dt = data_dr['dt']
# the drives are already in GHz
qubitI = data_dr['QubitI']
qubitQ = data_dr['QubitQ']
cavI = data_dr['CavityI']
cavQ = data_dr['CavityQ']
qubitI_rev = data_dr_rev['QubitI']
qubitQ_rev = data_dr_rev['QubitQ']
cavI_rev = data_dr_rev['CavityI']
cavQ_rev = data_dr_rev['CavityQ']

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

Utot_rev = np.kron(np.eye(qdim),np.eye(cdim))
for j in range(len(qubitI)):
    Udt = expm(-1j*(H0+qubitI_rev[j]*Hc[2]+qubitQ_rev[j]*Hc[3]+cavI_rev[j]*Hc[0]+cavQ_rev[j]*Hc[1]))
    Utot_rev = Udt@Utot_rev

D = 6
u = (fock(cdim,0)+fock(cdim,4)).unit()
G_tar = ket2dm(u)[0:D,0:D]

Utotcav = Utot[0:cdim, 0:cdim]
Gtot = Utotcav.T.conj()@Utotcav
G = Gtot[0:D,0:D]#the truncated U from simulation

cost = np.sum(np.abs(G_tar-G)**2)/D**2
print(f'cost for Um is {cost}')

U_rev = Utot_rev[0:D,0:D]
U_tar = Utot.T.conj()[0:D,0:D]

cost = np.sum(np.abs(U_tar-U_rev)**2)/D**2
print(f'cost for Ur is {cost}')

#%%State collapse
xmax = 2.5#for Wigner plots
xaxis = np.linspace(-xmax, xmax, 21)

#cavity state initial
rd1 = ket2dm(coherent(cdim, 0)).full()
# rd1 = R4gg_c_qobj
# rd1 = R4ge_c_qobj
# rd1 = rho_final.full()

def Dop(alpha, a):
    return expm(alpha*a.T.conj()-np.conjugate(alpha)*a)
amp = -1#1-1j#
# rd1 = Dop(amp,c)@RR@Dop(amp,c).T.conj()
# rd1 = thermal_dm(cdim,0.15).full()
# rd1 = Dop(0.2,c)@rd1@Dop(0.2,c).T.conj()

plot_wigner(Qobj(rd1), xvec=xaxis, yvec=xaxis)

#qubit projectors
gproj = np.kron(ug@ug.T.conj(),np.eye(cdim))
eproj = np.kron(ue@ue.T.conj(),np.eye(cdim))

#initial qubit-cavity state
R0 = np.kron(ug@ug.T.conj(), rd1)

#1st operation by Um
R1 = Utot@R0@Utot.T.conj()

#qubit measurement
pg = np.real( np.trace(gproj@R1) )
pe = np.real( np.trace(eproj@R1) )#1-pg
print(f'<p_overlap> is {pg}')

#state if qubit is in g
R2g = gproj@R1@gproj/pg
# R2g_qobj = Qobj(R2g, dims = [[qdim, cdim],[qdim,cdim]])
# R2g_c_qobj = R2g_qobj.ptrace(1)
# R2g_c_qobj = R2g[0:cdim, 0:cdim]
# plot_wigner(Qobj(R2g_c_qobj), alpha_max = 3)

#state if qubit is in e
R2e = eproj@R1@eproj/pe

#Apply the 2nd operation (reverse)
R3g = Utot_rev@R2g@Utot_rev.T.conj()
# R3g_qobj = Qobj(R3g, dims = [[qdim, cdim],[qdim,cdim]])
# R3g_c_qobj = R3g_qobj.ptrace(1)
# plot_wigner(Qobj(R3g_c_qobj), alpha_max = 3)

R3e = Utot_rev@R2e@Utot_rev.T.conj()


plot_wigner(Qobj(R3g[0:D,0:D]), xvec=xaxis, yvec=xaxis)
Fi = fidelity(Qobj(ket2dm(u)[0:D,0:D]).unit(),Qobj(R3g[0:D,0:D]).unit())**2



print(f'pq1g is {pg}')
print(f'pq1e is {pe}')
print(f'Fidelity to target is {Fi}')

