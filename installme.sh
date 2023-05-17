                         
# installing conda
mkdir ./Executables
if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    if [ "$(uname -m)" == "x86_64" ]; then
        wget -O ./Executables/Miniconda3-latest.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    elif [ "$(uname -m)" == "arm64" ]; then
        wget -O ./Executables/Miniconda3-latest.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    fi
elif [ "$(uname)" == "Linux" ]; then
    # Do something under Linux platform
    wget -O ./Executables/Miniconda3-latest.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
bash ./Executables/Miniconda3-latest.sh -b  -p ./Executables/miniconda -f


# create your own virtual environment in a new folder
source ./Executables/miniconda/bin/activate
python -m venv ./Executables/py_wireDAQ
source ./Executables/py_wireDAQ/bin/activate


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
pip install NAFFlib
pip install dask

# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name py_wireDAQ --display-name "py_wireDAQ"
#========================================


# Install CERN packages
#=========================================
git clone https://gitlab.cern.ch/mrufolo/fillingstudies.git ./Executables/py_wireDAQ/fillingstudies
pip install -e ./Executables/py_wireDAQ/fillingstudies

# Install acc-py and NXCALS
python -m pip install git+https://gitlab.cern.ch/acc-co/devops/python/acc-py-pip-config.git
python -m pip install --no-cache nxcals 

git clone https://gitlab.cern.ch/lhclumi/lumi-followup.git ./Executables/py_wireDAQ/lumi-followup
pip install -e ./Executables/py_wireDAQ/lumi-followup/nx2pd
#=========================================                     


