'''
This file is solely meant for plotting of pulses in various ways. Made by KX because I
need some visuals to interpret stuff properly
'''

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from qutip import *

#####################################
#### Hot-Fix for QuTip Bloch Vector ######
#####################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    '''
    The Arrow3D function for the Bloch vector plotting in
    Qutip has been busted for a long time now, so this portion is to
    override the internal Qutip's Matplotlib dependencies

    To note: One day I want to add labels for the arrows
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

class Bloch(Bloch):
    '''
    Made a new Bloch class to fix that add_vectors function that
    doesn't work in QuTiP's Bloch()
    '''
    #def __init__(self, *args):
    #    super().__init__(*args)

    def plot_vectors(self,):
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.vectors)):

            xs3d = self.vectors[k][1] * np.array([0, 1])
            ys3d = -self.vectors[k][0] * np.array([0, 1])
            zs3d = self.vectors[k][2] * np.array([0, 1])

            color = self.vector_color[np.mod(k, len(self.vector_color))]

            if self.vector_style == '':
                # simple line style
                self.axes.plot(xs3d, ys3d, zs3d,
                               zs=0, zdir='z', label='Z',
                               lw=self.vector_width, color=color)
            else:
                # decorated style, with arrow heads
                a = Arrow3D(xs3d, ys3d, zs3d,
                            mutation_scale=self.vector_mutation,
                            lw=self.vector_width,
                            arrowstyle=self.vector_style,
                            color=color)

                self.axes.add_artist(a)

    def plot_axes_labels(self):
        # axes labels
        opts = {'fontsize': self.font_size,
                'color': self.font_color,
                'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)

        for a in (self.axes.xaxis.get_ticklines() +
                  self.axes.xaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.yaxis.get_ticklines() +
                  self.axes.yaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.zaxis.get_ticklines() +
                  self.axes.zaxis.get_ticklabels()):
            a.set_visible(False)

#############################################
#############################################
#############################################

# Helper function
def dm2bloch(dm, ground_state=0,):
    '''
    Function to convert the qubit (or qutrit) density matrix to the relevant qubit bloch vector
    '''
    if type(dm )is not np.ndarray:
        dm = np.array(dm)
    dm = dm[ground_state:ground_state+2, ground_state:ground_state+2]
    
    u = np.real(2*dm[0,1])
    v = np.imag(2*dm[1,0])
    w = np.real(2*dm[0,0] - 1)

    return [u,v,w]

###################################

def show_wigner(sim_result, plot_range = 6, resolution = 60):
    '''
    Function to plot the Wigner Quasiprobability Functions of the initial and final cavity states

    Parameters
    --------------
    sim_result : Result object
        The results of time evolution of the state under Qutip's solvers

    plot_range : int
        The limit of the wigner function to plot. Note that this value should cover the entire range of
        what you want to visualise in the plot. Generally, cdims is sufficient.

    resolution : int
        The number of pixels on each side of the plot
    '''
    # plot wigner function
    max_range = plot_range
    displ_array = np.linspace(-max_range, max_range, resolution + 1)

    # create first plot
    fig, axes = plt.subplots(1,2)
    fig = plt.figure(figsize = (20,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #ax1.set_aspect('equal', 'box')
    #ax2.set_aspect('equal', 'box')
    wigner_f0 = wigner(sim_result.states[0].ptrace(0), displ_array, displ_array)
    wigner_f1 = wigner(sim_result.states[-1].ptrace(0), displ_array, displ_array)
    cont0 = ax1.pcolormesh(displ_array, displ_array, wigner_f0, cmap = "bwr")
    cont1 = ax2.pcolormesh(displ_array, displ_array, wigner_f1, cmap = "bwr")
    #cb = fig.colorbar(cont0)

    cont0.set_clim(-1/np.pi, 1/np.pi)
    cont1.set_clim(-1/np.pi, 1/np.pi)

    ax1.set_title("Initial State", fontsize = 20)
    ax2.set_title("Final State", fontsize = 20)

    plt.show()

def show_bloch(sim_result, ground_state = 0, excited_state = 1):
    '''
    Function to 3D-plot the Bloch sphere of the Qubit state

    Parameters
    -------------
    sim_result : Result object
        The results of time evolution of the state under Qutip's solvers
        
    ground_state : int (Default = 0)
        The density matrix energy level for the qubit ground state. The excited
        state is defined as ground_state + 1
    '''
    # Define the figure and axes
    fig = plt.figure(figsize = (20,8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plotting the Bloch sphere
    b = Bloch(fig, ax1)
    b.vector_color = ['b']
    b.make_sphere()

    b.add_vectors([
        dm2bloch(np.array(sim_result.states[0].ptrace(1))),
        ])
    
    b.render()
    ax1.set_title("Initial State", fontsize = 20)

    b = Bloch(fig, ax2)
    b.vector_color = ['b']
    b.make_sphere()

    b.add_vectors([
        dm2bloch(np.array(sim_result.states[-1].ptrace(1))),
        ])
    
    b.render()
    ax2.set_title("Final State", fontsize = 20)


    #fig.suptitle("Qubit Evolution", fontsize = 24)
    plt.show()


def show_evolution(sim_results, plot_range = 6, resolution = 60, frame_step = 4, savepath = ''):
    '''
    A function to show the time evolution of the qubit cavity system

    Parameters
    -------------
    sim_results : Result object
        The results of time evolution of the state under Qutip's solvers

    plot_range : float or int
        The limits of the wigner function plot

    resolution : int
        The number of grid cells for the wigner

    frame_step : int
        The interval between frames

    savepath : string
        The output directory to save to
    '''
    if savepath == '':
        savepath = "C:\\Users\\Lee Kai Xiang\\Documents\\Work\\PC4199 Honours Project\\Codebase\\New Grape\\"
    
    # plot wigner function
    max_range = plot_range
    displ_array = np.linspace(-max_range, max_range, resolution + 1)
    wigner_list = [wigner(x.ptrace(0), displ_array, displ_array) for x in sim_results.states[::frame_step]]
    bloch_list = [x.ptrace(1) for x in sim_results.states[::frame_step]]

    # create first plot
    fig, axes = plt.subplots(1,2)
    fig = plt.figure(figsize = (20,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    #fig.tight_layout()
    
    cont0 = ax1.pcolormesh(displ_array, displ_array, wigner_list[0], cmap = "bwr")
    cb = fig.colorbar(cont0)

    # refresh function for wigner
    def animated_wigner(frame):
        wigner = wigner_list[frame]
        cont0 = ax1.pcolormesh(displ_array, displ_array, wigner, cmap = "bwr")
        cont0.set_clim(-1/np.pi, 1/np.pi)

        ax1.set_title("Cavity State", fontsize = 20)

    # refresh function for bloch
    def animated_bloch(frame):
        bloch = bloch_list[frame]

        b = Bloch(fig,ax2)
        b.vector_color = ['b']
        b.make_sphere()

        b.add_vectors([dm2bloch(bloch)])
        b.render()
        
        ax2.set_title("Qubit State", fontsize = 20)

    def refresh(frame):
        animated_bloch(frame)
        animated_wigner(frame)

    anim = FuncAnimation(fig, refresh, frames=len(wigner_list), interval=100)
    anim.save(savepath + '.gif', writer='Pillow')
