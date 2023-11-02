#**************************
# FILE: heat2D_python_mpi4py.py
# RUN: mpirun -n <NUMTASKS> python heat2D_python_mpi4py.py <ITERATIONS> <XDIM> <YDIM>
# DESCRIPTIONS:
# HEAT2D Example - Parallelized C Version
# This example is based on a simplified two-dimensional heat
# equation domain decomposition.  The initial temperature is computed to be
# high in the middle of the domain and zero at the boundaries.  The
# boundaries are held at zero throughout the simulation.  During the
# time-stepping, an array containing two domains is used; these domains
# alternate between old data and new data.
# In this parallelized version, the grid is decomposed by the master
# process and then distributed by rows to the worker processes.  At each
# time step, worker processes must exchange border data with neighbors,
# because a grid point's current temperature depends upon it's previous
# time step value plus the values of the neighboring grid points.  Upon
# completion of all time steps, the worker processes return their results
# to the master process.
#**************************/
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys

#***************************
#  function inidat
#***************************/
def inidat(nx,ny,u):
    for ix in range(nx):
        for iy in range(ny):
            u[0,ix,iy] = ix * (nx - ix - 1) * iy * (ny - iy - 1)

#**************************
# function update
#**************************/
def update(start, end, ny, u1, u2):
    temp = 1-2.0*(Cx+Cy)
    for ix in range(start,end+1):
        for iy in range(1,ny-1):
            u2[ix,iy] = u1[ix,iy]*temp + Cx * (u1[ix+1,iy] + u1[ix-1,iy]) + Cy * (u1[ix,iy+1] + u1[ix,iy-1])
#**************************/

MAXWORKER  = 17          # maximum number of worker tasks
MINWORKER  = 1          # minimum number of worker tasks
BEGIN      = 1          # message tag
LTAG       = 2          # message tag
RTAG       = 3          # message tag
NONE       = 0          # indicates no neighbour
DONE       = 4          # message tag
MASTER     = 0          # taskid of first process

Cx = 0.1          # blend factor in heat equation
Cy = 0.1          # blend factor in heat equation

def main( PROGNAME, STEPS, NXPROB, NYPROB ):

    u = np.zeros((2,NXPROB,NYPROB))        # array for grid

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
        # print("Grid size: X= %d  Y= %d  Time steps= %d" % (NXPROB,NYPROB,STEPS))
        inidat(NXPROB, NYPROB, u)

    # Distribute work to workers.  Must first figure out how many rows to
    # send and what to do with extra rows.
        averow = NXPROB//numworkers
        extra = NXPROB%numworkers
        offset = 0

        initial_time = MPI.Wtime()
        for i in range(1,numworkers+1):
            rows = averow
            if i <= extra:
                rows+=1

        # Tell each worker who its neighbors are, since they must exchange
        # data with each other.
            if i == 1:
                above = NONE
            else:
                above = i - 1
            if i == numworkers:
                below = NONE
            else:
                below = i + 1

        # Now send startup information to each worker
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above, dest=i, tag=BEGIN)
            comm.send(below, dest=i, tag=BEGIN)
            comm.Send(u[0,offset:offset+rows,:], dest=i, tag=BEGIN)
            offset += rows

        # Now wait for results from all worker tasks
        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv([u[0,offset,:],rows*NYPROB,MPI.DOUBLE], source=i, tag=DONE)

        final_time = MPI.Wtime()
        print("Grid size: ({},{}) Iterations: {} Workers: {} Time: {:9.6f} s".format(NXPROB,NYPROB,STEPS,numworkers,final_time-initial_time))
    # End of master code
    
    #********* workers code ************/
    elif taskid != MASTER:
        # Array is already initialized to zero - including the borders
        # Receive my offset, rows, neighbors and grid partition from master
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)
        comm.Recv([u[0,offset,:],rows*NYPROB,MPI.DOUBLE], source=MASTER, tag=BEGIN)

        # Determine border elements.  Need to consider first and last columns.
        # Obviously, row 0 can't exchange with row 0-1.  Likewise, the last
        # row can't exchange with last+1.
        start=offset
        end=offset+rows-1
        if offset==0:
            start=1
        if (offset+rows)==NXPROB:
            end-=1

        # Begin doing STEPS iterations.  Must communicate border rows with
        # neighbours.  If I have the first or last grid row, then I only need
        # to  communicate with one neighbour
        iz = 0;
        for it in range(STEPS):
            if above != NONE:
                req=comm.Isend([u[iz,offset,:],NYPROB,MPI.DOUBLE], dest=above, tag=RTAG)
                comm.Recv([u[iz,offset-1,:],NYPROB,MPI.DOUBLE], source=above, tag=LTAG)
            if below != NONE:
                req=comm.Isend([u[iz,offset+rows-1,:],NYPROB,MPI.DOUBLE], dest=below, tag=LTAG)
                comm.Recv([u[iz,offset+rows,:],NYPROB,MPI.DOUBLE], source=below, tag=RTAG)
            # Now call update to update the value of grid points
            update(start,end,NYPROB,u[iz],u[1-iz]);
            iz = 1 - iz

        # Finally, send my portion of final results back to master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
        comm.Send([u[iz,offset,:],rows*NYPROB,MPI.DOUBLE], dest=MASTER, tag=DONE)

#********* input code ***********/
if _name_ == '_main_':
    if int(len(sys.argv)) == 4:
        PROGNAME = sys.argv[0]
        STEPS = int(sys.argv[1])
        NXPROB = int(sys.argv[2])
        NYPROB = int(sys.argv[3])
        main(PROGNAME, STEPS, NXPROB, NYPROB)
    else:
        print("Usage: {} <ITERATIONS> <XDIM> <YDIM>".format(sys.argv[0]))