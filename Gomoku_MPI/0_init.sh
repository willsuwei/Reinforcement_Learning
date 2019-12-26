rm -rf ./venv
    virtualenv --system-site-packages -p python3 ./venv

source ./venv/bin/activate

python -m pip install --upgrade pip
pip install tensorlayer==1.11.1
pip install tensorflow==1.15
pip install pygame==2.0.0.dev6
# pip install mpi4py==2.0.0
# pip install --upgrade numpy
