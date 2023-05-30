import numpy as np
import pandas as pd
from pathlib import Path
import importlib
import dask.dataframe as dd
import gc

gc.enable()


import matplotlib.pyplot as plt



# Local imports
#------------------------------------------------
import WireDAQ.PandasPlus           # Make sure this import is after pandas
import WireDAQ.Constants as cst
import WireDAQ.NXCALS as nx
import WireDAQ.Parser as parser
import WireDAQ.Efficiency as eff

main = __import__('000_Efficiency_per_fill')


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
_default_fig_width  = 2000
_default_fig_height = 400

_default_device = 'DBLM'

_default_import = 'local'
_default_path   = '/home/lumimod/work/run/data/2023/rawdata/'
_default_out    = '/eos/user/p/phbelang/www/Monitoring_BBCW/'


_default_path = '/home/phbelang/002mount'

#------------------------------------------------

# display(pd.DataFrame(gc.get_stats()))


def obj_size_fmt(num):
    if num<10**3:
        return "{:.2f}{}".format(num,"B")
    elif ((num>=10**3)&(num<10**6)):
        return "{:.2f}{}".format(num/(1.024*10**3),"KB")
    elif ((num>=10**6)&(num<10**9)):
        return "{:.2f}{}".format(num/(1.024*10**6),"MB")
    else:
        return "{:.2f}{}".format(num/(1.024*10**9),"GB")


def memory_usage():
    memory_usage_by_variable=pd.DataFrame({k:sys.getsizeof(v)\
    for (k,v) in globals().items()},index=['Size'])

    memory_usage_by_variable=memory_usage_by_variable.T
    memory_usage_by_variable=memory_usage_by_variable.sort_values(by='Size',ascending=False).head(10)
    memory_usage_by_variable['Size']=memory_usage_by_variable['Size'].apply(lambda x: obj_size_fmt(x))
    return memory_usage_by_variable

import psutil
def print_RAM():
        # Getting % usage of virtual_memory ( 3rd field)
    print(40*'-' + '\nAfter')
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


print_RAM()


FILL = 8773
data_path= _default_path
fill     = FILL
    
# STARTS HERE
#=================================================   


# Finding filling pattern
#-------------------------------------------------
bb_df_b1,bb_df_b2 = parser.fill_filling_pattern(fill=FILL,data_path= _default_path,n_LR = 21)
#-------------------------------------------------



# Declaring master bin times
#-------------------------------------------------
dt = 30
unix_s,unix_e = parser.fill_unix_times(FILL,data_path=data_path)
unix_bins     = np.arange(unix_s,unix_e,dt/1e-9)
#-------------------------------------------------

# Import and bin dBLM 
#-------------------------------------------------
variables = []
for beam in beams:
    variables += beam['dBLM_Amp'].values()

dblm = parser.from_parquet2bin(fill=FILL,variables = variables,bins=unix_bins,beamMode = None,data_path= _default_path)
#-------------------------------------------------


# Import lumi and compute total lumi
#-------------------------------------------------
variables  = [LHC.bb_Luminosity['ATLAS'],LHC.bb_Luminosity['CMS']]
df_lumi     = parser.from_parquet(fill=FILL,variables = variables,beamMode = None,data_path= _default_path)
df_lumi_tot = eff.compute_lumi_tot(df_lumi,experiments = ['ATLAS','CMS'])

#-------------------------------------------------

# Computing Efficiency
#-------------------------------------------------
# droping near-zero values
n_colliding   = len(bb_df_b1['HO partner in ATLAS'].dropna())
avg_per_bunch = df_lumi_tot['bb_LUMI_TOT'].apply(lambda line:np.sum(line)/n_colliding)
noise_level   = 0.00115
ROI_idx       = avg_per_bunch[avg_per_bunch>10/2000].index[[0,-1]]

_tmp = pd.concat([dblm,df_lumi_tot]).sort_index().loc[ROI_idx[0]:ROI_idx[1]]

df_eff = eff.compute_dBLM_efficiency(_tmp,beams)
#----------------------------------



# Adding extra data for analysis
#-------------------------------------------------
LHC_vars   = LHC._getVarList(subset=[   'Xing',
                                        'Fill',
                                        'Bmode',
                                        'betastar'])


beams_vars = sum([beam._getVarList(subset=[ 'Intensity',
                                            'Nb']) for beam in beams],[])
wires_vars = sum([wire._getVarList(subset=[ 'I']) for wire in wires['B2']],[])
variables = sorted(LHC_vars+beams_vars+wires_vars)

df_extra = parser.from_parquet(fill=FILL,variables = variables,beamMode = None,data_path= _default_path)

# Intensity
#-------------------------------------------------
variables = sum([beam._getVarList(subset=['bb_Intensity']) for beam in beams],[])
df_intensity = parser.from_parquet2bin(fill=FILL,variables = variables,bins=unix_bins,beamMode = None,data_path= _default_path)



database = pd.concat([df_extra,df_intensity,df_lumi,df_eff])
database = parser.make_timestamps(database)



