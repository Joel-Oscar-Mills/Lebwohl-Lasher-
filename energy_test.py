import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys

#=======================================================================
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


#=======================================================================
def block_energy(rows, angles, energies, parity):
    """
    """
    for ix in range(1,rows+1):

        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-angles[ix+1,parity::2] ))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-angles[ix-1,parity::2] ))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-np.roll(angles[ix,:],-1)[(parity)::2] ))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-np.roll(angles[ix,:],1)[(parity)::2] ))**2
        parity = 1 - parity


#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en


MAXWORKER  = 3          # maximum number of worker tasks
MINWORKER  = 1          # minimum number of worker tasks
BEGIN      = 1          # message tag
DONE       = 2          # message tag
ATAG       = 3          # message tag
BTAG       = 4          # message tag
NONE       = 0          # indicates no neighbour
MASTER     = 0          # taskid of first process


def main(PROGNAME, NMAX):

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
        energies = np.zeros((NMAX,NMAX))
        serial_energies = np.zeros((NMAX,NMAX))

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
            comm.Recv([energies[offset,:],rows*NMAX,MPI.DOUBLE], source=i, tag=DONE)

        # Compute one site energies as in the serial algorithm
        for ix in range(NMAX):
            for iy in range(NMAX):
                serial_energies[ix,iy] = one_energy(angles,ix,iy,NMAX)

        final_time = MPI.Wtime()
        print(energies-serial_energies)

    # End of master code
    
    
    #********* workers code ************/
    elif taskid != MASTER:
        # Array is already initialized to zero 
        # Receive my offset, rows & neighbors 
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)

        # set aside the exact amount of memory this process requires to receive \
        # the master's portion with room for neighbours above & below
        angles = np.zeros((rows+2,NMAX))
        energies = np.zeros((rows,NMAX))
        comm.Recv([angles[1,:],rows*NMAX,MPI.DOUBLE], source=MASTER, tag=BEGIN)

        #if taskid == 1:
            #print(angles[1])
            #print(np.roll(angles[1],1))

        # Must communicate border rows with neighbours. 
        req=comm.Isend([angles[1,:],NMAX,MPI.DOUBLE], dest=above, tag=ATAG)
        req=comm.Isend([angles[rows,:],NMAX,MPI.DOUBLE], dest=below, tag=BTAG)
        comm.Recv([angles[0,:],NMAX,MPI.DOUBLE], source=above, tag=BTAG)
        comm.Recv([angles[rows+1,:],NMAX,MPI.DOUBLE], source=below, tag=ATAG)


        # Now compute the energies
        parity = offset%2
        block_energy(rows,angles,energies,parity)
        parity = 1 - offset%2
        block_energy(rows,angles,energies,parity)

        # Finally, send my portion of final results back to master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
        comm.Send([energies[0,:],rows*NMAX,MPI.DOUBLE], dest=MASTER, tag=DONE)



#********* input code ***********/
if __name__ == '__main__':
    if int(len(sys.argv)) == 2:
        PROGNAME = sys.argv[0]
        NMAX= int(sys.argv[1])
        main(PROGNAME, NMAX)
    else:
        print("Usage: {} <NMAX>".format(sys.argv[0]))