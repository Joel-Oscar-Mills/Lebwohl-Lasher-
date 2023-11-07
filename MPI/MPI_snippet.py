import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys


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


def block_energy(start, end, NMAX, angles, energies):
    """
    """
    for ix in range(start,end+1):

        energies[ix,:] += 0.5 - 1.5*(np.cos(angles[ix,:]-angles[(ix+1)%NMAX,:]))**2
        energies[ix,:] += 0.5 - 1.5*(np.cos(angles[ix,:]-angles[(ix-1)%NMAX,:]))**2
        energies[ix,:] += 0.5 - 1.5*(np.cos(angles[ix,:]-np.roll(angles[ix,:],-1)))**2
        energies[ix,:] += 0.5 - 1.5*(np.cos(angles[ix,:]-np.roll(angles[ix,:],1)))**2


MAXWORKER  = 3          # maximum number of worker tasks
MINWORKER  = 1          # minimum number of worker tasks
BEGIN      = 1          # message tag
DONE       = 2          # message tag
ATAG       = 3          # message tag
BTAG       = 4          # message tag
NONE       = 0          # indicates no neighbour
MASTER     = 0          # taskid of first process


def main(PROGNAME, NMAX):

        # Create and initialise lattice of angles and energies
    angles = np.zeros((NMAX,NMAX))
    energies = np.zeros((NMAX,NMAX))

    # First, find out my taskid and how many tasks are running
    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1

    #********* master code ***********/
    if taskid == MASTER:
    # Check if numworkers is within range - quit if not
        if (numworkers > MAXWORKER) or (numworkers < MINWORKER):
            print("ERROR: the number of tasks must be between %d and %d." % (MINWORKER+1,MAXWORKER+1))
            print("Quitting...")
            comm.Abort()

        #print("Starting mpi_heat2D with %d worker tasks." % numworkers)

        # Initialize grid
        angles = initdat(NMAX)

        # Distribute work to workers.  Must first figure out how many rows to
        # send and what to do with extra rows.
        averow = NMAX//numworkers
        extra = NMAX%numworkers
        offset = 0

        initial_time = MPI.Wtime()
        for i in range(1,numworkers+1):
            rows = averow
            if i <= extra:
                rows+=1

            # Tell each worker who its neighbors are, since they must exchange
            # data with each other.
            if i == 1:
                above = numworkers
            else:
                above = i - 1
            if i == numworkers:
                below = 1
            else:
                below = i + 1

            # Now send startup information to each worker
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above, dest=i, tag=BEGIN)
            comm.send(below, dest=i, tag=BEGIN)
            comm.Send(angles[offset:offset+rows,:], dest=i, tag=BEGIN)
            offset += rows

             # Now wait for results from all worker tasks
        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv([angles[offset,:],rows*NMAX,MPI.DOUBLE], source=i, tag=DONE)



    elif taskid != MASTER:
        # Array is already initialized to zero 
        # Receive my offset, rows, neighbors and grid partition from master
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)
        comm.Recv([angles[offset,:],rows*NMAX,MPI.DOUBLE], source=MASTER, tag=BEGIN)
        print("above = ", above)
        print("below = ", below)

        # Determine border elements.  
        start=offset
        end=offset+rows-1

        # Must communicate border rows with neighbours. 
        req=comm.Isend([angles[offset,:],NMAX,MPI.DOUBLE], dest=above, tag=ATAG)
        req=comm.Isend([angles[offset+rows-1,:],NMAX,MPI.DOUBLE], dest=below, tag=BTAG)
        comm.Recv([angles[(offset-1)%NMAX,:],NMAX,MPI.DOUBLE], source=above, tag=BTAG)        
        comm.Recv([angles[(offset+rows)%NMAX,:],NMAX,MPI.DOUBLE], source=below, tag=ATAG)
        
        print(angles)

        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
        comm.Send([angles[offset,:],rows*NMAX,MPI.DOUBLE], dest=MASTER, tag=DONE)
        
    print("complete ", taskid)


#********* input code ***********/
if __name__ == '__main__':
    if int(len(sys.argv)) == 2:
        PROGNAME = sys.argv[0]
        NMAX= int(sys.argv[1])
        main(PROGNAME, NMAX)
    else:
        print("Usage: {} <NMAX>".format(sys.argv[0]))