#!/bin/bash
#
#SBATCH --job-name=training_gpu
#SBATCH --output=training_gpu.log
#
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=47:00
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-user=wei.su@sjsu.edu
#SBATCH --mail-type=END
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# srun -p gpu --gres=gpu --pty /bin/bash

source ./venv/bin/activate
mpiexec --oversubscribe -np 3 python -u train_mpi.py

