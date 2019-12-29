srun -n 1 -N 1 -c 14 --pty /bin/bash

source ./venv/bin/activate
mpiexec --oversubscribe -np 10 python -u train_mpi.py
