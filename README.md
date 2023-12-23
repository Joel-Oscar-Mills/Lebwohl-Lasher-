We simulate liquid crystals in the Lebwohl-Lasher model using the Metropolis-Hastings Algorithm. 
Several ways to parallelize/accelerate the simulation are explored:

  - MPI (checkerboard parallelization)
  - Numba (prange)
  - Cythonization
  - Numpy Vectorization (used in all approaches)
