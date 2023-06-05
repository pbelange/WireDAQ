import numpy as np
import pandas as pd
from pathlib import Path
import gc
import subprocess
import time


# BOKEH
import bokeh.plotting as bk
import bokeh.models as bkmod
import bokeh.layouts as bklay

# Local imports
#------------------------------------------------
import WireDAQ.PandasPlus           # Make sure this import is after pandas
import WireDAQ.Constants as cst
import WireDAQ.NXCALS as nx
import WireDAQ.Parser as parser
import WireDAQ.Efficiency as eff


# Creating NXCALS variable containers
#------------------------------------------------
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
LHC       = nx.NXCALSLHC()
b_slots   = np.arange(3564)
#------------------------------------------------


# Setting default values
#------------------------------------------------
_default_device = 'DBLM'
_default_path   = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/'
# _default_HTML   = '/eos/user/p/phbelang/www/Monitoring_BBCW/DBLM/'
_default_efficiency_HTML   = '/eos/user/p/phbelang/www/Monitoring_efficiency/DBLM/'
_default_bbbsignature_HTML = '/eos/user/p/phbelang/www/Monitoring_bbbsignature/DBLM/'


# Importing fill metadata
#==========================================
df_list = []
for state in ['on', 'off']:
    _df = pd.read_pickle(f'filter_wires_{state}.pkl')
    _df.insert(0, 'Wires', state.upper())
    df_list.append(_df)
df = pd.concat(df_list).sort_index()
#==========================================


# Running analysis for fills
for fill in df.index:

    try:
        # Creating Efficiency parquet
        #==========================================
        checkfile = f"{_default_path}/efficiency_data/{_default_device}/FILL{fill}.parquet"
        if not Path(checkfile).exists():
            subprocess.run(["python", "001_dBLM_efficiency.py", f"{fill}"   ,
                                                                "-p"        , f"{_default_path}",
                                                                "-e"        , f"{_default_path}/efficiency_data/{_default_device}/"])
        #------------------------------------------
        gc.collect()
        time.sleep(2)
        gc.collect()

        # Creating HTML file
        #==========================================
        checkfile = f"{_default_efficiency_HTML}/FILL{fill}.html"
        if not Path(checkfile).exists():
            subprocess.run(["python", "002_efficiency_HTML.py", f"{fill}" ,
                                                            "-p"    , f"{_default_path}",
                                                            "-d"    , f"{_default_device}",
                                                            "-e"    , f"{_default_efficiency_HTML}"])
        #------------------------------------------

        # Creating HTML file
        #==========================================
        checkfile = f"{_default_bbbsignature_HTML}/FILL{fill}.html"
        if not Path(checkfile).exists():
            subprocess.run(["python", "002_bbbsignature_HTML.py", f"{fill}" ,
                                                            "-p"    , f"{_default_path}",
                                                            "-d"    , f"{_default_device}",
                                                            "-e"    , f"{_default_bbbsignature_HTML}"])
        #------------------------------------------
        gc.collect()

    except Exception as error:
        # handle the exception
        print(f"FILL {fill} | An exception occurred:", error)
