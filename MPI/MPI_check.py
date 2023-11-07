import mpi4py
import sys
import numpy as np
from mpi4py import MPI

def initdat(NMAX):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to fill the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  NONE
    """
    angles = np.random.random_sample((NMAX,NMAX))*2.0*np.pi
    return angles

def main(PROGNAME,NMAX):

    comm = MPI.COMM_WORLD
    worker = comm.Get_rank()
    size = comm.Get_size()

    angles = initdat(NMAX)
    print(angles)
    

if __name__ == '__main__':
    PROGNAME = sys.argv[0]
    NMAX = int(sys.argv[1])
    main(PROGNAME,NMAX)