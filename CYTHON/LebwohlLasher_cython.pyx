# cython: language_level=3
"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from libc.math cimport cos
from math import ceil
import cython
cimport numpy as cnp
from cython.parallel import prange
cimport openmp
from cython cimport Py_ssize_t

@cython.boundscheck(False)
@cython.wraparound(False) 

#=======================================================================
def initdat(long nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(cnp.ndarray[cnp.float64_t, ndim=2] angles,cnp.ndarray[cnp.float64_t, ndim=2] energies,int pflag,long NMAX):
    """
    Arguments:
	  angles (float(nmax,nmax)) = array that contains lattice angles;
      energies (float(nmax,nmax)) = array that contains lattice energies;
	  pflag (int) = parameter to control plotting;
      NMAX (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    cdef cnp.ndarray[cnp.float64_t, ndim=2] u = np.cos(angles)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] v = np.sin(angles)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.arange(NMAX)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y = np.arange(NMAX)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] cols = np.zeros((NMAX,NMAX))

    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        cols = energies
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = angles%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(angles)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*NMAX)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(cnp.ndarray[cnp.float64_t, ndim=2] arr,int nsteps,double Ts,double runtime,cnp.ndarray[cnp.float64_t, ndim=1] ratio,cnp.ndarray[cnp.float64_t, ndim=2] energy,cnp.ndarray[cnp.float64_t, ndim=2] order,long NMAX):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(NMAX,NMAX),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
cdef void row_energy(double[:, :] angles, double[:, :, :] energies, Py_ssize_t ix, long NMAX, int parity, int new) nogil:
    """
    Directly manipulate memory views for angles and energies.
    """
    # Assuming angles is a 2D array and energies is a 3D array
    cdef Py_ssize_t j
    cdef double angle_diff

    # Loop through the selected elements based on parity
    for j in range(parity, NMAX, 2):
        # Calculating the periodic boundary conditions manually
        angle_diff = angles[ix, j] - angles[(ix + 1) % NMAX, j]
        energies[new, ix, j] += 0.5 - 1.5 * cos(angle_diff) ** 2

        angle_diff = angles[ix, j] - angles[(ix - 1 + NMAX) % NMAX, j]  # Add NMAX for negative wrapping
        energies[new, ix, j] += 0.5 - 1.5 * cos(angle_diff) ** 2

        # For the roll, you will have to manually implement the roll behavior
        # since np.roll isn't available without the GIL.
        angle_diff = angles[ix, j] - angles[ix, (j - 1 + NMAX) % NMAX]
        energies[new, ix, j] += 0.5 - 1.5 * cos(angle_diff) ** 2

        angle_diff = angles[ix, j] - angles[ix, (j + 1) % NMAX]
        energies[new, ix, j] += 0.5 - 1.5 * cos(angle_diff) ** 2

    parity = 1 - parity

#=======================================================================
def get_Q(cnp.ndarray[cnp.float64_t, ndim=2] angles,long NMAX):
    """
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=3] field = np.vstack((np.cos(angles),np.sin(angles))).reshape(2,NMAX,NMAX)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Q = 1.5*np.einsum("aij,bij->ab",field,field)/(NMAX**2) - 0.5*np.eye(2,2)
    return Q
#=======================================================================
def get_order(cnp.ndarray[cnp.float64_t, ndim=3] Q,int STEPS):
    """
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] order = np.zeros(STEPS)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eigenvalues = np.zeros(2)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] eigenvectors = np.zeros((2,2))
    cdef Py_ssize_t t
    for t in range(STEPS):
        eigenvalues,eigenvectors = np.linalg.eig(Q[t])
        order[t] = eigenvalues.max()
    return order
#=======================================================================
def MC_substep(cnp.ndarray[cnp.float64_t, ndim=2] angles,cnp.ndarray[cnp.float64_t, ndim=2] rangles,cnp.ndarray[cnp.float64_t, ndim=3] energies,int parity,double Ts,long NMAX,cnp.ndarray[cnp.float64_t, ndim=1] E,cnp.ndarray[cnp.float64_t, ndim=3] Q,cnp.ndarray[cnp.float64_t, ndim=1] R,int it,int threads):
    """
    """
    # Compute the energies
    cdef Py_ssize_t ix
    for ix in prange(NMAX, nogil=True, num_threads=threads):
        row_energy(angles, energies, ix, NMAX, parity, 0)
    parity = (parity + NMAX%2)%2

    # Perturb the angles randomly (for on-parity sites)
    if parity == 0:
        angles[parity::2,parity::2] += rangles[parity::2,parity::2]
        angles[(1-parity)::2,(1-parity)::2] += rangles[(1-parity)::2,(1-parity)::2]
    else:
        angles[parity::2,(1-parity)::2] += rangles[parity::2,(1-parity)::2]
        angles[(1-parity)::2,parity::2] += rangles[(1-parity)::2,parity::2]
   
    # Compute new energies
    for ix in prange(NMAX, nogil=True, num_threads=threads):
        row_energy(angles, energies, ix, NMAX, parity, 1)
    parity = (parity + NMAX%2)%2

    # Determine which changes to accept (store as Boolean arrays)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] guaranteed = (energies[1] <= energies[0])
    cdef cnp.ndarray[cnp.float64_t, ndim=2] boltz = np.exp(-(energies[1]-energies[0])/Ts)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] accept = guaranteed + (1-guaranteed)*(boltz >= np.random.uniform(0.0,1.0,size=(NMAX,NMAX)))

    # Adjust energies based on which sites were accepted
    energies[1] = accept*energies[1] + (1-accept)*energies[0]
   
    if parity == 0:
        # Record this rank's portion of the total acceptance ratio
        R[it] += (np.sum(accept[parity::2,parity::2])+np.sum(accept[1-parity::2,1-parity::2]))/(NMAX**2)

        # Undo the rejected changes
        angles[parity::2,parity::2] -= (1-accept[parity::2,parity::2])*rangles[parity::2,parity::2]
        angles[(1-parity)::2,(1-parity)::2] -= (1-accept[(1-parity)::2,(1-parity)::2])*rangles[(1-parity)::2,(1-parity)::2]
    else:
        # Record this rank's portion of the total acceptance ratio
        R[it] += (np.sum(accept[parity::2,(1-parity)::2])+np.sum(accept[(1-parity)::2,parity::2]))/(NMAX**2)

        # Undo the rejected changes
        angles[parity::2,(1-parity)::2] -= (1-accept[parity::2,(1-parity)::2])*rangles[parity::2,(1-parity)::2]
        angles[(1-parity)::2,parity::2] -= (1-accept[(1-parity)::2,parity::2])*rangles[(1-parity)::2,parity::2]
    
#=======================================================================
def MC_step(cnp.ndarray[cnp.float64_t, ndim=2] angles,double Ts,long NMAX,cnp.ndarray[cnp.float64_t, ndim=1] E,cnp.ndarray[cnp.float64_t, ndim=3] Q,cnp.ndarray[cnp.float64_t, ndim=1] R,Py_ssize_t it,int threads):
    """
    """
    cdef double scale=0.1+Ts
    cdef cnp.ndarray[cnp.float64_t, ndim=2] rangles = np.random.normal(scale=scale, size=(NMAX,NMAX))
    cdef cnp.ndarray[cnp.float64_t, ndim=3] energies = np.zeros((2,NMAX,NMAX))

    # Perform the MC step for all on-parity sites, then for off-parity sites
    cdef int parity = 0
    MC_substep(angles,rangles,energies,parity,Ts,NMAX,E,Q,R,it,threads)
    parity = 1
    MC_substep(angles,rangles,energies,parity,Ts,NMAX,E,Q,R,it,threads)

    # Recompute on-parity site energies as these will have changed after the off-parity update
    parity = 0
    cdef Py_ssize_t ix
    for ix in prange(NMAX, nogil=True, num_threads=threads):
        row_energy(angles, energies, ix, NMAX, parity, 1)

    E[it] = np.sum(energies[1])
    Q[it] = get_Q(angles,NMAX)

    return angles, energies, E, Q, R
#=======================================================================
def simulation_runtime(int STEPS,long NMAX,double Ts,int pflag, int threads):
    """
    """
    # Create arrays to store energy, acceptance ratio and averaged Q matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=1] E = np.zeros(STEPS)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] R = np.zeros(STEPS)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] Q = np.zeros((STEPS,2,2))

    # Initialize grid
    cdef cnp.ndarray[cnp.float64_t, ndim=2] angles = initdat(NMAX)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] energies = np.zeros((NMAX,NMAX))

    initial_time = openmp.omp_get_wtime()

    cdef Py_ssize_t it
    for it in range(STEPS):
        angles, energies, E, Q, R = MC_step(angles,Ts,NMAX,E,Q,R,it,threads)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] order = get_order(Q,STEPS)
    final_time = openmp.omp_get_wtime()
    runtime = final_time-initial_time

    return runtime

#=======================================================================
def main(program):
    """
    """
    cdef int NP = 4
    cdef cnp.ndarray[long, ndim=1] list_NMAX = np.array([ceil(10*(10**(2*i/20))) for i in range(21)])
    cdef int STEPS = 50
    cdef double Ts = 0.5
    cdef pflag = 2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] runtimes = np.zeros(21)
    cdef long NMAX = 0

    cdef Py_ssize_t i
    for i in range(len(list_NMAX)):
        NMAX = list_NMAX[i]
        runtimes[i] = simulation_runtime(STEPS,NMAX,Ts,pflag,NP)

    np.savetxt(f"./cython_runtimes_vs_NMAX_{NP}.txt",runtimes)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if _name_ == '_main_':
    if int(len(sys.argv)) == 1:
        main(sys.argv[0])
    else:
        print("Usage: python {}".format(sys.argv[0]))
#=======================================================================
