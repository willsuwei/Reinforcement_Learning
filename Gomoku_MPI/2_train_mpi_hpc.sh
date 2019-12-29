srun -p gpu --gres=gpu --pty /bin/bash

source ./venv/bin/activate
mpiexec --oversubscribe -np 3 python -u train_mpi.py
