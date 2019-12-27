source ./venv/bin/activate

mpiexec -np 6 python -u train_mpi.py
