                                              
# Download acc models
git clone https://gitlab.cern.ch/acc-models/acc-models-lhc.git

# Copying relevant optics files over ssh:
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6 py_wireDAQ/optics


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
#git clone --single-branch --branch feature/wire_3.0 https://github.com/pbelange/lhcmask.git py_wireDAQ/lhcmask
git clone https://github.com/lhcopt/lhcmask.git py_wireDAQ/lhcmask
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

git clone https://github.com/PyCOMPLETE/FillingPatterns.git py_wireDAQ/FillingPatterns
pip install -e py_wireDAQ/FillingPatterns


# Additionnal packages
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

# Removing the installer
rm acc-py-2020.11-installer.sh


# Modifying xtrack for python 3.7.9 compatibility
sed -i "s|{i_aper_1=}, {i_aper_0=}|i_aper_1={i_aper_1}, i_aper_0={i_aper_0}|" py_wireDAQ/xtrack/xtrack/loss_location_refinement/loss_location_refinement.py
sed -i "s|{presence_shifts_rotations=}|presence_shifts_rotations={presence_shifts_rotations}|" py_wireDAQ/xtrack/xtrack/loss_location_refinement/loss_location_refinement.py
sed -i "s|{iteration=}|iteration={iteration}|" py_wireDAQ/xtrack/xtrack/loss_location_refinement/loss_location_refinement.py


# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name py_wireDAQ --display-name "py_wireDAQ"