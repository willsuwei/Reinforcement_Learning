source ./venv/bin/activate

mpiexec --oversubscribe -np 3 python -u train_mpi.py
