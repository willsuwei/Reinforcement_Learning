rm -rf ./venv
    virtualenv --system-site-packages -p python3 ./venv

brew install mpich
brew install openmpi

source ./venv/bin/activate

python -m pip install --upgrade pip
pip install tensorlayer==1.11.1
pip install tensorflow==1.15
pip install pygame==2.0.0.dev6
pip install mpi4py==3.0.3
pip install filelock
# pip install --upgrade numpy
