
import numpy as np
import pandas as pd
import dask.dataframe as dd
import gc


# Local imports
#------------------------------------------------
import WireDAQ.PandasPlus           # Make sure this import is after pandas
import WireDAQ.Constants as cst
import WireDAQ.NXCALS as nx
import WireDAQ.Spark as spark

import sys
sys.path.append('./Executables/py_wireDAQ/fillingstudies/')
from tools_box.analysis import bb_tool_box as bbsc 
#------------------------------------------------


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

_default_import = 'local'

_default_path   = '/home/lumimod/work/run/data/2023/rawdata/'
# _default_path   = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/rawdata/'

_default_out    = '/eos/user/p/phbelang/www/Monitoring_BBCW/'
#------------------------------------------------



#==================================================
def fill_unix_times(fill,data_path=_default_path):
    _df = dd.read_parquet(data_path + f'/HX:FILLN={fill}',columns=[LHC.Fill]).compute()
    _df = _df.sort_index()
    return _df.index[0],_df.index[-1]
#==================================================

    

#=====================================================
# Downloading data from NXCALS
#=====================================================
def from_NXCALS(start_time=None,end_time=None,fill=None,variables = None):
    # Minimal variables to be used
    #============================================
    if variables is None:
        LHC_vars   = LHC._getVarList(subset=[   'bb_Luminosity',
                                                'Luminosity',
                                                'Xing',
                                                'Fill',
                                                'Bmode',
                                                'betastar'])
        LHC_vars.remove(LHC.bb_Luminosity['LHCB'])
        LHC_vars.remove(LHC.bb_Luminosity['ALICE'])

        beams_vars = sum([beam._getVarList(subset=[ 'bb_Intensity',
                                                    'bb_Intensity_B',
                                                    'Intensity',
                                                    'Intensity_B',
                                                    'Nb',
                                                    'dBLM_Amp']) for beam in beams],[])


        wires_vars = sum([wire._getVarList(subset=[ 'I']) for wire in wires['B2']],[])
        variables = sorted(LHC_vars+beams_vars+wires_vars)
    #============================================


    # Loading the data
    #============================================
    spark_session = spark.SparkSession()
    database      = spark_session.query(start_time=start_time,end_time=end_time,fill=fill,variables = variables)
    #============================================

    return database


#=====================================================
# Downloading data from EOS pre-saved data
#=====================================================
def from_EOS(fill=None,variables = None,beamMode = None,DATA_PATH = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/rawdata/'):
    # Minimal variables to be used
    #============================================
    if variables is None:
        LHC_vars   = LHC._getVarList(subset=[   'bb_Luminosity',
                                                'Luminosity',
                                                'Xing',
                                                'Fill',
                                                'Bmode',
                                                'betastar'])
        LHC_vars.remove(LHC.bb_Luminosity['LHCB'])
        LHC_vars.remove(LHC.bb_Luminosity['ALICE'])

        beams_vars = sum([beam._getVarList(subset=[ 'bb_Intensity',
                                                    'bb_Intensity_B',
                                                    'Intensity',
                                                    'Intensity_B',
                                                    'Nb',
                                                    'dBLM_Amp']) for beam in beams],[])


        wires_vars = sum([wire._getVarList(subset=[ 'I']) for wire in wires['B2']],[])
        variables = sorted(LHC_vars+beams_vars+wires_vars)
    #============================================


    # Importing the data
    #============================================
    if beamMode is None:
        aux = dd.read_parquet(DATA_PATH + f'HX:FILLN={fill}',columns=variables)
    else:
        aux = dd.read_parquet(DATA_PATH + f'HX:FILLN={fill}' + f'/HX:BMODE={beamMode.upper()}',columns=variables)
        
    database = aux.compute()
    #============================================

    # Adding proper timestamp
    #============================================
    database = database.sort_index()
    database.index.name = 'unix'
    database.insert(0,'Timestamp',database.index)
    database.insert(1,'Time',1e-9*(database.index - database.index[0]))
    database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(cst.TZONE))
    #============================================

    return database


#====================================================
# make_timestamps
#====================================================
def make_timestamps(df):
    # Adding proper timestamp
    #-----------------------------
    df = df.sort_index()
    df.index.name = 'unix'
    df.insert(0,'Timestamp',df.index)
    df.insert(1,'Time',1e-9*(df.index - df.index[0]))
    df['Timestamp'] = df['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(cst.TZONE).tz_localize(None))
    #-----------------------------
    return df


#=====================================================
# BIN DATA with loading
#=====================================================
def from_parquet(fill=None,file = None,variables = None,beamMode = None,data_path= _default_path):

    # Importing the data
    #-----------------------------
    if fill is not None:
        if beamMode is None:
            _partition = dd.read_parquet(data_path + f'/HX:FILLN={fill}',columns=variables)
        else:
            _partition = dd.read_parquet(data_path + f'/HX:FILLN={fill}'+ f'/HX:BMODE={beamMode.upper()}',columns=variables)
    elif file is not None:
        _partition = dd.read_parquet(file,columns=variables)
    #-----------------------------

    df        = _partition.compute()
    df        = df.sort_index()

    # Removing raw data
    #-----------------------------
    del(_partition)
    gc.collect()
    #-----------------------------

    # Cleaning up
    #-----------------------------
    df = df.sort_index()
    df.index.name = 'unix'
    #-----------------------------

    return df

def from_parquet2bin(fill=None,file = None,variables = None,rename=None,dt = 60,bins=None,beamMode = None,data_path= _default_path):
    
    if bins is None:
        unix_s,unix_e = fill_unix_times(fill,data_path=data_path)
        unix_bins     = np.arange(unix_s,unix_e,dt/1e-9)
    else:
        unix_bins = bins


    # Importing the data
    #-----------------------------
    if fill is not None:
        if beamMode is None:
            _partition = dd.read_parquet(data_path + f'/HX:FILLN={fill}',columns=variables)
        else:
            _partition = dd.read_parquet(data_path + f'/HX:FILLN={fill}'+ f'/HX:BMODE={beamMode.upper()}',columns=variables)
    elif file is not None:
        _partition = dd.read_parquet(file,columns=variables)
    #-----------------------------

    _df        = _partition.compute()
    _df        = _df.sort_index()

    # Binning
    #-----------------------------
    per_var_list = []
    for col in _df.columns:

        observable = col
        per_var_list.append(_df.bin_unix(observable,bins=unix_bins))
    #-----------------------------

    # Appending
    df = pd.concat(per_var_list,axis=1)

    # Removing raw data
    #-----------------------------
    del(_partition)
    del(_df)
    gc.collect()
    #-----------------------------

    # Cleaning up
    #-----------------------------
    df = df.sort_index()
    df.index.name = 'unix'
    if rename is not None:
        df.rename(columns=rename,inplace=True)
    #-----------------------------
    return df



#=====================================================
# BIN DATA with loading
#=====================================================
from pathlib import Path
def load_and_bin_sequential(fill=None,variables = None,dt = 60,beamMode = None,data_path= _default_path):
    unix_s,unix_e = fill_unix_times(fill,data_path=data_path)
    unix_bins     = np.arange(unix_s,unix_e,dt/1e-9)


    per_beam_list = []
    for beam in beams:
        # if beam.name == 'B1':
        #     continue
        
        # Variables for this beam
        variables = [beam['dBLM_Amp']['V'],
                    beam['dBLM_Amp']['H-V-S']]
        
        # Iterate through bmode
        per_mode_list = []
        for bmode_path in Path(data_path + f'/HX:FILLN={fill}').glob("*"):
            # just for testing
            # if 'STABLE' not in str(bmode_path):
            #     continue

            
            _partition = dd.read_parquet(bmode_path,columns=variables)
            _partition['unix'] = _partition.index
            _partition = _partition.sort_values(by='unix')
            _partition = _partition.set_index('unix')

            _df = _partition.compute()
        
            per_type_list = []
            for dblmType in ['V','H-V-S']:
                observable = beam.dBLM_Amp[dblmType]
                per_type_list.append(_df.bin_unix(observable,bins=unix_bins))

            # Appending
            per_mode_list.append(pd.concat(per_type_list,axis=1))

            del(_partition)
            del(_df)
            gc.collect()

        # Appending
        per_beam_list.append(pd.concat(per_mode_list,axis=0))

    #Appending
    df = pd.concat(per_beam_list,axis=1)

    # Adding proper timestamp
    #============================================
    df = df.sort_index()
    df.index.name = 'unix'
    df.insert(0,'Timestamp',df.index)
    df.insert(1,'Time',1e-9*(df.index - df.index[0]))
    df['Timestamp'] = df['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(cst.TZONE))
    #============================================
    return df


#=====================================================
def get_train_lengths(schedule, ring_length_slots = len(b_slots)):
    # TAKEN FROM FILLING PATTERN: https://github.com/PyCOMPLETE/FillingPatterns.git

    # Recreate the pattern
    pattern = np.zeros(ring_length_slots,dtype=bool).astype(int)
    pattern[schedule.index] = 1

    # Identify trains
    jumps = np.diff(pattern)
    start_trains = np.where(jumps==1)[0]+1
    end_trains = np.where(jumps==-1)[0]+1
    if end_trains[0]<start_trains[0]:
        start_trains = np.array([0] + list(start_trains))

    # Lenghts of trains and gaps
    return end_trains - start_trains
#=====================================================

#=====================================================
def fill_filling_pattern(fill=None,data_path= _default_path,n_LR = 21):


    variables = [beam['bb_Intensity'] for beam in beams]
    # qerying data every 10 minutes just to find filling pattern
    _df       = from_parquet2bin(fill=fill,variables = variables,dt=10*60,data_path= data_path)

    # Extracting pattern from intensity
    #=====================================================
    filled_b_slots = {}
    for beam in beams:
        
        # Only looking at after max of intensity
        bb_I    = _df[beam['bb_Intensity']].dropna()
        max_loc = bb_I.apply(lambda line: np.sum(line)).argmax()
        bb_I    = bb_I.iloc[max_loc:]


        # Taking bb intensity just after max signal and then 1/4 of the way
        probe1 = bb_I.iloc[2-1:2+1].mean(axis=0)
        probe2 = bb_I.iloc[len(bb_I)//4-1:len(bb_I)//4+1].mean(axis=0)


        assert(np.all((probe1!=0) == (probe2!=0))), 'Error on the filling scheme'

        filled_b_slots[beam.name] = (probe1!=0).astype(int)
    #=====================================================

    # Computing schedule
    #=====================================================
    bb_df_b1,bb_df_b2 = bbsc.bbschedule(filled_b_slots['B1'],filled_b_slots['B2'],numberOfLRToConsider = n_LR, 
                                                                                    Dict_Detectors = {  'ATLAS' : int(0/8 * (len(b_slots) * 2)),
                                                                                                        'ALICE' : int(1/8 * (len(b_slots) * 2)),
                                                                                                        'CMS'   : int(4/8 * (len(b_slots) * 2)),
                                                                                                        'LHCB'  : int(7/8 * (len(b_slots) * 2)) - 3 })


    # Removing raw data
    #-----------------------------
    del(bb_I)
    del(_df)
    gc.collect()
    #-----------------------------

    return bb_df_b1,bb_df_b2
#=====================================================





#=====================================================
# MEMORY USAGE
#=====================================================
class Memory_profiler():
    def __init__(self,):
    
        from memory_profiler import profile
        
        #functions to be profiled
        self.fill_unix_times    = profile(fill_unix_times)
        self.from_NXCALS        = profile(from_NXCALS)
        self.from_EOS           = profile(from_EOS)
        self.getFillingPattern  = profile(getFillingPattern)
        self.get_train_lengths  = profile(get_train_lengths)
        self.load_and_bin       = profile(load_and_bin)
        self.load_and_bin_sequential = profile(load_and_bin_sequential)



    def Memory_profiler(self,):
        return self