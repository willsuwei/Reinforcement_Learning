source ./venv/bin/activate

mpiexec -np 6 python -u train_mpi.py

# mpiexec --oversubscribe -np 8 python -u train_mpi.py
