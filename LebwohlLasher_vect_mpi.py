import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI

MAXWORKER  = 3          # maximum number of worker tasks
MINWORKER  = 1          # minimum number of worker tasks
BEGIN      = 1          # message tag
DONE       = 2          # message tag
ATAG       = 3          # message tag
BTAG       = 4          # message tag
NONE       = 0          # indicates no neighbour
MASTER     = 0          # taskid of first process

#=======================================================================
def initdat(NMAX):
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
    arr = np.random.random_sample((NMAX,NMAX))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(angles,energies,pflag,NMAX):
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
    u = np.cos(angles)
    v = np.sin(angles)
    x = np.arange(NMAX)
    y = np.arange(NMAX)
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
def block_energy(rows, angles, energies, parity):
    """
    """
    for ix in range(1,rows+1):

        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-angles[ix+1,parity::2] ))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-angles[ix-1,parity::2] ))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-np.roll(angles[ix,:],-1)[(parity)::2] ))**2
        energies[ix-1,parity::2] += 0.5 - 1.5*(np.cos( angles[ix,parity::2]-np.roll(angles[ix,:],1)[(parity)::2] ))**2
        parity = 1 - parity

    # Reset parity to initial value
    parity = (parity + rows%2)%2

#=======================================================================
def partial_Q(angles,NMAX):
    """
    """
    (dimx, dimy) = np.shape(angles)
    field = np.vstack((np.cos(angles),np.sin(angles))).reshape((2,dimx,dimy))
    Q = 1.5*np.einsum("aij,bij->ab",field,field)/(NMAX**2) - 0.5*(dimx*dimy/(NMAX**2))*np.eye(2,2)
    return Q
#=======================================================================
def get_order(Q,STEPS):
    """
    """
    order = np.zeros(STEPS)
    for t in range(STEPS):
        eigenvalues,eigenvectors = np.linalg.eig(Q[t])
        order[t] = eigenvalues.max()
    return order
#=======================================================================
def MC_substep(comm,angles,rangles,energies,Ts,NMAX,rows,parity,above,below,R,it):

    # Must communicate border rows with neighbours. 
    req=comm.Isend([angles[1,:],NMAX,MPI.DOUBLE], dest=above, tag=ATAG)
    req=comm.Isend([angles[rows,:],NMAX,MPI.DOUBLE], dest=below, tag=BTAG)
    comm.Recv([angles[0,:],NMAX,MPI.DOUBLE], source=above, tag=BTAG)
    comm.Recv([angles[rows+1,:],NMAX,MPI.DOUBLE], source=below, tag=ATAG)

    # Now compute the energies
    block_energy(rows,angles,energies[0],parity)


    # Perturb the angles randomly (for on-parity sites)
    if parity == 0:
        angles[(1+parity):rows+1:2,parity::2] += rangles[parity::2,parity::2]
        angles[(2-parity):rows+1:2,(1-parity)::2] += rangles[(1-parity)::2,(1-parity)::2]
    else:
        angles[(1+parity):rows+1:2,(1-parity)::2] += rangles[parity::2,(1-parity)::2]
        angles[(2-parity):rows+1:2,parity::2] += rangles[(1-parity)::2,parity::2]


    # Must communicate new border rows with neighbours.
    req=comm.Isend([angles[1,:],NMAX,MPI.DOUBLE], dest=above, tag=ATAG)
    req=comm.Isend([angles[rows,:],NMAX,MPI.DOUBLE], dest=below, tag=BTAG)
    comm.Recv([angles[0,:],NMAX,MPI.DOUBLE], source=above, tag=BTAG)
    comm.Recv([angles[rows+1,:],NMAX,MPI.DOUBLE], source=below, tag=ATAG)

    # Compute new energies
    block_energy(rows,angles,energies[1],parity)

    # Determine which changes to accept (store as Boolean arrays)
    guaranteed = (energies[1] <= energies[0])
    boltz = np.exp(-(energies[1]-energies[0])/Ts)
    accept = guaranteed + (1-guaranteed)*(boltz >= np.random.uniform(0.0,1.0,size=(rows,NMAX)))

    # Adjust energies based on which sites were accepted
    energies[1] = accept*energies[1] + (1-accept)*energies[0]

    if parity == 0:
        # Record this rank's portion of the total acceptance ratio
        R[it] += (np.sum(accept[parity::2,parity::2])+np.sum(accept[1-parity::2,1-parity::2]))/(NMAX**2)

        # Undo the rejected changes
        angles[(1+parity):rows+1:2,parity::2] -= (1-accept[parity::2,parity::2])*rangles[parity::2,parity::2]
        angles[(2-parity):rows+1:2,(1-parity)::2] -= (1-accept[(1-parity)::2,(1-parity)::2])*rangles[(1-parity)::2,(1-parity)::2]
    else:
        # Record this rank's portion of the total acceptance ratio
        R[it] += (np.sum(accept[parity::2,(1-parity)::2])+np.sum(accept[(1-parity)::2,parity::2]))/(NMAX**2)

        # Undo the rejected changes
        angles[(1+parity):rows+1:2,(1-parity)::2] -= (1-accept[parity::2,(1-parity)::2])*rangles[parity::2,(1-parity)::2]
        angles[(2-parity):rows+1:2,parity::2] -= (1-accept[(1-parity)::2,parity::2])*rangles[(1-parity)::2,parity::2]


#=======================================================================
def MC_step(comm,angles,energies,Ts,NMAX,rows,offset,above,below,E,Q,R,it):
    """
    """
    scale=0.1+Ts
    rangles = np.random.normal(scale=scale, size=(rows,NMAX))
    energies = np.zeros((2,rows,NMAX))

    parity = offset%2
    MC_substep(comm,angles,rangles,energies,Ts,NMAX,rows,parity,above,below,R,it)

    parity = 1 - offset%2
    MC_substep(comm,angles,rangles,energies,Ts,NMAX,rows,parity,above,below,R,it)

    # Recompute on-parity site energies as these will have changed after the off-parity update
    block_energy(rows,angles,energies[1],offset%2)
    E[it] = np.sum(energies[1])
    Q[it] = partial_Q(angles,NMAX)

    return angles, energies, E, Q, R

#=======================================================================
def main(program, STEPS, NMAX, Ts, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """

    # First, find out my taskid and how many tasks are running
    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1
  
    # Create arrays to store energy, acceptance ratio and averaged Q matrix
    E = np.zeros(STEPS)
    R = np.zeros(STEPS)
    Q = np.zeros((STEPS,2,2))


    #********* master code ***********/
    if taskid == MASTER:
    # Check if numworkers is within range - quit if not
        if (numworkers > MAXWORKER) or (numworkers < MINWORKER):
            print("ERROR: the number of tasks must be between %d and %d." % (MINWORKER+1,MAXWORKER+1))
            print("Quitting...")
            comm.Abort()

        # Initialize grid
        angles = initdat(NMAX)
        energies = np.zeros((NMAX,NMAX))

        # Plot initial frame of lattice
        plotdat(angles,energies,pflag,NMAX)

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

        temp_E = np.zeros(STEPS)
        temp_Q = np.zeros((STEPS,2,2))
        temp_R = np.zeros(STEPS)
        # Now wait for results from all worker tasks
        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv([angles[offset,:],rows*NMAX,MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([energies[offset,:],rows*NMAX,MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_E[0:],STEPS,MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_Q[0:],STEPS*4,MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_R[0:],STEPS,MPI.DOUBLE], source=i, tag=DONE)
            E += temp_E
            Q += temp_Q
            R += temp_R

        order = get_order(Q,STEPS)
        final_time = MPI.Wtime()
        runtime = final_time-initial_time

        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, NMAX,STEPS,Ts,order[STEPS-1],runtime))
        #savedat(angles,STEPS,Ts,runtime,R,E,order,NMAX)
        plotdat(angles,energies,pflag,NMAX)
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
        energies = np.zeros((2,rows,NMAX))
        comm.Recv([angles[1,:],rows*NMAX,MPI.DOUBLE], source=MASTER, tag=BEGIN)

        for it in range(STEPS):
            angles, energies, E, Q, R = MC_step(comm,angles,energies,Ts,NMAX,rows,offset,above,below,E,Q,R,it)
            
        # Finally, send my portion of final results back to master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
        comm.Send([angles[1,:],rows*NMAX,MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([energies[1,:,:],rows*NMAX,MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([E[0:],STEPS,MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([Q[0:],STEPS*4,MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([R[0:],STEPS,MPI.DOUBLE], dest=MASTER, tag=DONE)



#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
