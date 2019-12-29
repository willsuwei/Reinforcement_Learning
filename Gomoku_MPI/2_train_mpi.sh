source ./venv/bin/activate

mpiexec -np 3 python -u train_mpi.py

# mpiexec --oversubscribe -np 8 python -u train_mpi.py
