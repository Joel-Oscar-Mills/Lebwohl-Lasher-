Simulation of liquid crystals in the Lebwohl-Lasher model using the Metropolis-Hastings Algorithm. 
Several ways to parallelize/accelerate the simulation are explored (all approaches use Numpy vectorization):
  - MPI (checkerboard parallelization)
  - Numba (prange)
  - Cythonization
