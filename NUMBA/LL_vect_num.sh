#!/bin/bash

#SBATCH --job-name=Lebwohl_Lasher
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:0:0
#SBATCH --mem-per-cpu=10000M
#SBATCH --account=phys026162

## Direct output to the following files.
## (The %j is replaced by the job id.)
#SBATCH -e skeleton_err_%j.txt
#SBATCH -o skeleton_out_%j.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Just in case this is not loaded already...
module load languages/intel/2020-u4

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}" 
printf "\n\n"

# Submit
numba LL_vect_num_many.py

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"