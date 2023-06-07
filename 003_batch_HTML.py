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
_default_path   = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/'

_default_efficiency_HTML   = '/eos/user/p/phbelang/www/Monitoring_efficiency/'
_default_bbbsignature_HTML = '/eos/user/p/phbelang/www/Monitoring_bbbsignature/'


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
for fill in [8685]:#df.index:
    for device in ['DBLM','BCTF']:
        try:
            # Creating Efficiency parquet
            #==========================================
            checkfile = f"{_default_path}/efficiency_data/{device}/FILL{fill}.parquet"
            if not Path(checkfile).exists():
                subprocess.run(["python", f"001_{device}_efficiency.py", f"{fill}"   ,
                                                                    "-p"        , f"{_default_path}",
                                                                    "-e"        , f"{_default_path}/efficiency_data/{device}/"])
            #------------------------------------------
            gc.collect()
            time.sleep(2)
            gc.collect()

            # Creating HTML file
            #==========================================
            checkfile = f"{_default_efficiency_HTML}/{device}/FILL{fill}.html"
            if not Path(checkfile).exists():
                subprocess.run(["python", "002_efficiency_HTML.py", f"{fill}" ,
                                                                "-p"    , f"{_default_path}",
                                                                "-d"    , f"{device}",
                                                                "-e"    , f"{_default_efficiency_HTML}/{device}/"])
            #------------------------------------------

            # Creating HTML file
            #==========================================
            checkfile = f"{_default_bbbsignature_HTML}/{device}/FILL{fill}.html"
            if not Path(checkfile).exists():
                subprocess.run(["python", "002_bbbsignature_HTML.py", f"{fill}" ,
                                                                "-p"    , f"{_default_path}",
                                                                "-d"    , f"{device}",
                                                                "-e"    , f"{_default_bbbsignature_HTML}/{device}/"])
            #------------------------------------------
            gc.collect()

        except Exception as error:
            # handle the exception
            print(f"FILL {fill}, {device} | An exception occurred:", error)
