                                              
# from http://bewww.cern.ch/ap/acc-py/installers/              
wget http://bewww.cern.ch/ap/acc-py/installers/acc-py-2020.11-installer.sh      
                      
bash ./acc-py-2020.11-installer.sh -p ./acc-py/base/2020.11/ -b    
           
# activate some acc-py python distribution:                 
source ./acc-py/base/2020.11/setup.sh       
                   
# create your own virtual environment in the folder "py_tn":       
acc-py venv py_wireDAQ       
                    
# activate your new environment    
source ./py_wireDAQ/bin/activate   
   
# and finish it with nxcals          
python -m pip install jupyterlab nxcals

# Add cpymad
pip install cpymad

# Add lhcmask and xsuite
git clone --single-branch --branch feature/wire_3.0 https://github.com/pbelange/lhcmask.git py_wireDAQ/lhcmask
pip install -e py_wireDAQ/lhcmask

git clone https://github.com/xsuite/xobjects py_wireDAQ/xobjects
pip install -e py_wireDAQ/xobjects

git clone https://github.com/xsuite/xdeps py_wireDAQ/xdeps
pip install -e py_wireDAQ/xdeps

git clone https://github.com/xsuite/xpart py_wireDAQ/xpart
pip install -e py_wireDAQ/xpart

git clone https://github.com/xsuite/xtrack py_wireDAQ/xtrack
pip install -e py_wireDAQ/xtrack

git clone https://github.com/xsuite/xfields py_wireDAQ/xfields
pip install -e py_wireDAQ/xfields

# Additionnal packages
pip install jupyterlab
pip install ipywidgets
pip install PyYAML
pip install pyarrow
pip install pandas
pip install matplotlib
pip install scipy
pip install ipympl

# Removing the installer
rm acc-py-2020.11-installer.sh