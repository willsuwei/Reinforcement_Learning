source ./venv/bin/activate

mpiexec --oversubscribe -np 8 python -u train_mpi.py
