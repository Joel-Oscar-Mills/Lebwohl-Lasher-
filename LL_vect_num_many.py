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
from math import ceil
from numba import njit, jit, prange

#=======================================================================
@njit
def initdat(nmax):
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
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
# Numba doesn't yet support randomized array generation \
# so create function in place
@njit
def rand(NMAX):
    noise = np.empty(NMAX**2,dtype=np.float64)
    for i in range(len(noise)):
        noise[i] = np.random.normal()
    return noise.reshape((NMAX,NMAX))
#=======================================================================
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,NMAX):
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
@njit
def row_energy(angles, energies, ix, NMAX, parity):
    """
    """
    energies[parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-angles[(ix+1)%NMAX,parity::2] ))**2
    energies[parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-angles[(ix-1)%NMAX,parity::2] ))**2
    energies[parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-np.roll(angles[ix,:],-1)[(parity)::2] ))**2
    energies[parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-np.roll(angles[ix,:],1)[(parity)::2] ))**2

#=======================================================================
# Numba does not support np.einsum() so a brute force approach is necessary
@njit
def get_Q(angles,NMAX):
    """
    """
    Qab = np.zeros((2,2))
    delta = np.eye(2,2)

    field = np.vstack((np.cos(angles),np.sin(angles))).reshape(2,NMAX,NMAX)
    for a in range(2):
        for b in range(2):
            for i in range(NMAX):
                for j in range(NMAX):
                    Qab[a,b] += 3*field[a,i,j]*field[b,i,j] - delta[a,b]
    Qab = Qab/(2*NMAX*NMAX)
    return Qab
#=======================================================================
@njit
def get_order(Q,STEPS):
    """
    """
    order = np.zeros(STEPS)
    for t in range(STEPS):
        eigenvalues,eigenvectors = np.linalg.eig(Q[t])
        order[t] = eigenvalues.max()
    return order
#=======================================================================
@njit(parallel=True)
def MC_substep(angles,rangles,energies,parity,Ts,NMAX,E,Q,R,it):
    """
    """
    # Compute the energies
    for ix in prange(NMAX):
        row_energy(angles, energies[0,ix,:], ix, NMAX, parity)
        parity = 1 - parity
    parity = (parity + NMAX%2)%2

    # Perturb the angles randomly (for on-parity sites)
    if parity == 0:
        angles[parity::2,parity::2] += rangles[parity::2,parity::2]
        angles[(1-parity)::2,(1-parity)::2] += rangles[(1-parity)::2,(1-parity)::2]
    else:
        angles[parity::2,(1-parity)::2] += rangles[parity::2,(1-parity)::2]
        angles[(1-parity)::2,parity::2] += rangles[(1-parity)::2,parity::2]
   
    # Compute new energies
    for ix in prange(NMAX):
        row_energy(angles, energies[1,ix,:], ix, NMAX, parity)
        parity = 1 - parity
    parity = (parity + NMAX%2)%2

    # Determine which changes to accept (store as Boolean arrays)
    guaranteed = (energies[1] <= energies[0])
    boltz = np.exp(-(energies[1]-energies[0])/Ts)
    accept = guaranteed + (1-guaranteed)*(boltz >= np.random.uniform(0.0,1.0,size=(NMAX,NMAX)))

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
@njit(parallel=True)
def MC_step(angles,Ts,NMAX,E,Q,R,it):
    """
    """
    scale=0.1+Ts
    rangles = rand(NMAX)
    energies = np.zeros((2,NMAX,NMAX))

    # Perform the MC step for all on-parity sites, then for off-parity sites
    parity = 0
    MC_substep(angles,rangles,energies,parity,Ts,NMAX,E,Q,R,it)
    parity = 1
    MC_substep(angles,rangles,energies,parity,Ts,NMAX,E,Q,R,it)

    # Recompute on-parity site energies as these will have changed after the off-parity update
    parity = 0
    for ix in prange(NMAX):
        row_energy(angles, energies[1,ix,:], ix, NMAX, parity)
        parity = 1 - parity

    E[it] = np.sum(energies[1])
    Q[it] = get_Q(angles,NMAX)

    return angles, energies, E, Q, R
#=======================================================================
def simulation_runtime(STEPS,NMAX,Ts,pflag):
    """
    """
    # Create arrays to store energy, acceptance ratio and averaged Q matrix
    E = np.zeros(STEPS)
    R = np.zeros(STEPS)
    Q = np.zeros((STEPS,2,2))

    # Initialize grid
    angles = initdat(NMAX)
    energies = np.zeros((NMAX,NMAX))

    initial_time = time.time()

    for it in range(STEPS):
        angles, energies, E, Q, R = MC_step(angles,Ts,NMAX,E,Q,R,it)

    order = get_order(Q,STEPS)
    final_time = time.time()
    runtime = final_time-initial_time
    return runtime

#=======================================================================
def main(program):
    """
    """
    list_NMAX = [ceil(8*(1000/8)**(i/20)) for i in range(21)]
     # Define command line arguments within main() as NUMBA does not support entering them in the terminal
    STEPS = 50
    Ts = 0.5
    pflag = 2
    runtimes = np.zeros(21)

    for i, NMAX in enumerate(list_NMAX):
        runtimes[i] = simulation_runtime(STEPS,NMAX,Ts,pflag)
     
    np.savetxt("./numba_runtime_vs_NMAX_8.txt",runtimes)

#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 1:
        PROGNAME = sys.argv[0]
        main(PROGNAME)
    else:
        print("Usage: python {}".format(sys.argv[0]))
#=======================================================================
