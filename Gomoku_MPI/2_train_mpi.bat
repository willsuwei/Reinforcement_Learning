setlocal
cd /d %~dp0

call activate tensorflow-gpu_1-15

mpiexec -np 10 python -u train_mpi.py
