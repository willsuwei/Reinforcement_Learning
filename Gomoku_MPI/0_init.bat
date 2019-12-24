call conda remove -y --name tensorflow-gpu_1-15 --all
call conda create -y -n tensorflow-gpu_1-15 pip python=3.7.1

call activate tensorflow-gpu_1-15

call python -m pip install --upgrade pip
call pip install tensorflow==1.15
call pip install tensorlayer==1.10.1
call pip install pygame==1.9.6
call pip install mpi4py
call pip install --upgrade numpy

pause
