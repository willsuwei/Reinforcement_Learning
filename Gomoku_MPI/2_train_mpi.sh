source ./venv/bin/activate

mpiexec -np 3 python -u train_mpi.py
