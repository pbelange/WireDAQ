                                  

# installing conda
mkdir ./Executables
wget -P ./Executables https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash ./Executables/Miniconda3-latest-Linux-x86_64.sh -b  -p ./Executables/miniconda -f


# create your own virtual environment in a new folder
source ./Executables/miniconda/bin/activate
python -m venv ./Executables/py_WireDAQ
source ./Executables/py_WireDAQ/bin/activate


# Install acc-py and NXCALS
python -m pip install git+https://gitlab.cern.ch/acc-co/devops/python/acc-py-pip-config.git
python -m pip install --no-cache nxcals 

# Install generic python packages
#========================================
pip install jupyterlab
pip install ipywidgets
pip install PyYAML
pip install pyarrow
pip install pandas
pip install matplotlib
pip install scipy
pip install ipympl
pip install ruamel.yaml
pip install rich
pip install lfm
pip install pynaff
pip install pynumdiff

# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name py_WireDAQ --display-name "py_WireDAQ"
#========================================


# Install CERN packages
#=========================================
pip install cpymad

git clone https://github.com/lhcopt/lhcmask.git ./Executables/py_WireDAQ/lhcmask
pip install -e ./Executables/py_WireDAQ/lhcmask

git clone https://github.com/xsuite/xobjects ./Executables/py_WireDAQ/xobjects
pip install -e ./Executables/py_WireDAQ/xobjects

git clone https://github.com/xsuite/xdeps ./Executables/py_WireDAQ/xdeps
pip install -e ./Executables/py_WireDAQ/xdeps

git clone https://github.com/xsuite/xpart ./Executables/py_WireDAQ/xpart
pip install -e ./Executables/py_WireDAQ/xpart

git clone https://github.com/xsuite/xtrack ./Executables/py_WireDAQ/xtrack
pip install -e ./Executables/py_WireDAQ/xtrack

git clone https://github.com/xsuite/xfields ./Executables/py_WireDAQ/xfields
pip install -e ./Executables/py_WireDAQ/xfields

git clone https://github.com/PyCOMPLETE/FillingPatterns.git ./Executables/py_WireDAQ/FillingPatterns
pip install -e ./Executables/py_WireDAQ/FillingPatterns
#=========================================
