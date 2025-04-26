#%%
import sys
sys.path.append(r"C:\Users\tanju\Dropbox\PY\OREO\grape")
sys.path.append(r"C:\Users\tanju\Dropbox\PY\OREO\grape\pygrape")
from pygrape import *
from qutip import *
from qutip.qip.device import Processor

import os
# Specify the directory in which you want to save the pulses
new_directory = r"C:\Users\tanju\Dropbox\PY\OREO"
os.chdir(new_directory)

from TK_basics import *

import numpy as np
import matplotlib.pyplot as plt
from pulse_plotting import *
import time
start_time = time.time()#checking how long the code takes

target_list = ["projbinomial04D10"] 
save_path_list = target_list

D = 10#truncation dimension
'''
One can choose a higher truncation dimension just to be more precise
'''
u = (fock(15,0)+fock(15,4)).unit()#state defined in higher dimension
Gtar = ket2dm(u)[0:D,0:D]#truncate

cost_f = 1e-5#the cost, element-wise mean square error defined in the paper
fid = 1-cost_f*D**2
'''
the code will optimise fid = 1-sum_{ij} |Gtar-G|_{ij}^2 (minimising sum_{ij} |Gtar-G|_{ij}^2)
Gtar is the target observable
G is the simulated observable given the pulses, G = 2<g|U_m^{dag}|g><g|U_m|g> - I, truncated to dimension D
So for a given cost_f, we plug in sum_{ij} |Gtar-G|_{ij}^2 = cost_f*D**2
'''
print(f'target fid is {fid}')

make_gif = 0                                 
args = dict(
    # Hamiltonian Parameters (in GHz)
    anharm = 0*-175.31e-3,
    kerr = -4.4e-6,#-3e-6,#-6e-6,
    chi_prime = -13.6e-6,
    chi = -1.482e-3,#-1.48e-3,
    qdim = 2,
    cdim = 15,
      
    # Pulse Parameters 
    num_points = 2000,
    init_drive_scale = 1e-3,
    dt = 1.0,# in ns
    
    # Penalties
    make_amp_cost_args = (1e-4, 1e-2),
    make_lin_amp_cost_args = None,
    make_deriv_cost_args = None,
    make_lin_deriv_cost_args = (1e2,),
      
    # max iterations
    discrepancy_penalty = 1e6,
    freq_range = None,
    shape_sigma = 800, 
    bound_amp = None,
    targ_fid = fid,#1 - 1e-3,

    # Approximations
    use_taylor = True,
    sparse = True,
    taylor_order = 20,
    
    # Type of pulse
    target = 'fock',
    param = 2,# Key parameter associated with target, e.g., displacement, etc.  
    )

def make_Hamiltonian(cdim, qdim, anharm = -0.0378, kerr = -1e-6,chi = -0.6e-3, chi_prime=-15.76e-6):
    # Operators
    q = tensor(destroy(qdim), qeye(cdim))
    qd = q.dag()
    c = tensor(qeye(qdim), destroy(cdim))
    cd = c.dag()

    # Hamiltonian
    H0 =  anharm/2 * qd*qd*q*q
    H0 += kerr/2 * cd*cd*c*c
    H0 += chi *cd*c*qd*q
    H0 += chi_prime / 2 * qd * q * cd * cd * c * c
    H0 *= 2 * np.pi

    # Control Hamiltonians
    Hc = [
        2*np.pi*(c + cd),
        1j*2*np.pi*(c - cd),
        2*np.pi*(q + qd),
        1j*2*np.pi*(q - qd),
        ]

    return H0, Hc

def make_setup(
    cdim, qdim, target = 'pi_pulse',
    anharm = 0, kerr = 0e-6, chi = 0e-3, chi_prime=0e-6,
    param = 0, use_taylor = False,
    sparse = False, taylor_order = 20,
    gauge_ops = None):
    
    H0, Hc = make_Hamiltonian(cdim, qdim, anharm = anharm, kerr=kerr, chi = chi, chi_prime=chi_prime)

    init, final = None, None

    setup = ObservableSetup_new_geonly_proj(
        H0, Hc,
        Gtar, cdim)
    
    return setup

def make_setups(**args):
    setup1 = make_setup(**args)

    args['cdim'] += 1
    setup2 = make_setup(**args)
    
    return [setup1, setup2]

if __name__ == "__main__":
    
    for i in range(len(save_path_list)):
        save_path = save_path_list[i]
        target = target_list[i]
        
        # Make the setups
        setups = make_setups(
            cdim = args['cdim'],
            qdim = args['qdim'],
            target = target,
            anharm = args['anharm'],
            kerr = args['kerr'],
            chi = args['chi'],
            chi_prime=args['chi_prime'],
            param = args['param'],
            use_taylor = args['use_taylor'],
            taylor_order = args['taylor_order'],
            sparse = args['sparse'],
            #gauge_ops = args['gauge_ops'],
            )
    
        # Initialise pulses
        num_ctrls = 2 if target == 'pi_pulse' or target == 'coherent' else 4
        # num_ctrls = 2 if target == 'pi_pulse' or target == 'coherent' or target == 'unitary' else 4
        init_ctrls = args['init_drive_scale'] * random_waves(num_ctrls, args['num_points'])
        
        #start with previous run
        # init_ctrls = np.zeros([num_ctrls, args['num_points']], dtype=np.float_)
        # init_ctrls[0,:] = cavI*1e-3
        # init_ctrls[1,:] = cavQ*1e-3
        # init_ctrls[2,:] = qubitI*1e-3
        # init_ctrls[3,:] = qubitQ*1e-3

        dt = args['dt']
    
        # Reporter functions
    
        pulse_names = ['CavityI', 'CavityQ','QubitI', 'QubitQ',]
    
        if target == 'pi_pulse': pulse_names = pulse_names[2:]
        # if target == 'coherent' or target == 'unitary': pulse_names = pulse_names[:2]
        
        reporters = [
                print_costs(),                                                    # Default, prints fidelities
                save_waves(pulse_names, 5),
                                                                                        # Saves the waves as ".npz"
                plot_waves(pulse_names, 5, iq_pairs = False),
                                                                                        # Plots the waves, saves as pdf
                # plot_trajectories(setups[0], 10),
                                                                                        # Plots probability trajectories
                # plot_states(10),
                plot_fidelity(10),                                               # Plots fidelity over iteration
                verify_from_setup(
                    make_setup(
                        cdim = args['cdim'],
                        qdim = args['qdim'],
                        target = target,
                        anharm = args['anharm'],
                        kerr = args['kerr'],
                        chi = args['chi'],
                        chi_prime = args['chi_prime'],
                        param = args['param'],
                        use_taylor = args['use_taylor'],
                        taylor_order = args['taylor_order'],
                        sparse = args['sparse'],
                        #gauge_ops = args['gauge_ops'],
                        ),
                                  10),
                                                                                        # Consistency check for higher fock trunctions
                save_script(__file__)
            ]
    
        # Penalty functions
        penalties = []
    
        if args['make_amp_cost_args'] != None:
            # To penalise amplitude to a soft threshold
            penalties += [make_amp_cost(*args['make_amp_cost_args'], iq_pairs=False),]
    
        if args['make_lin_amp_cost_args'] != None:
            # To penalise amplitude to a low number
            penalties += [make_lin_amp_cost(*args['make_lin_amp_cost_args'], iq_pairs=False),]
        
        if args['make_deriv_cost_args'] != None:
            # To penalise gradient for better bandwidth to a soft threshold
            penalties += [make_deriv_cost(*args['make_deriv_cost_args'], iq_pairs=False),]
        
        if args['make_lin_deriv_cost_args'] != None:
            # To penalise gradient for better bandwidth
            penalties += [make_lin_deriv_cost(*args['make_lin_deriv_cost_args'], iq_pairs=False),]
    
        # Additional Parameters
        opts = {
            "maxfun": 15000 * 5,
            "maxiter": 15000 * 5,
        }
    
    
        # Run grape
        result = run_grape(
            init_ctrls,
            setups,
            dt=dt,
            term_fid = args['targ_fid'],
            save_data = 10,
            reporter_fns = reporters,
            penalty_fns = penalties,
            discrepancy_penalty=args['discrepancy_penalty'],
            n_proc = 2,
            outdir = save_path,
            freq_range = args['freq_range'],
            shape_sigma = args['shape_sigma'],
            bound_amp = args['bound_amp'],
            #init_aux_params = args['init_aux_params'],
            **opts,
            )

print("")
print("--- %s seconds ---" % (time.time() - start_time))
