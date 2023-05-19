
import numpy as np
import pandas as pd


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
        beams_vars = sum([beam._getVarList(subset=[ 'bb_Intensity',
                                                    'bb_Intensity_B',
                                                    'Intensity',
                                                    'Intensity_B',
                                                    'Nb',
                                                    'dBLM_Amp']) for beam in beams],[])


        wires_vars = sum([wire._getVarList(subset=[ 'I',
                                                    'V']) for wire in wires['B1']+wires['B2']],[])
        variables = sorted(LHC_vars+beams_vars+wires_vars)
    #============================================


    # Loading the data
    #============================================
    spark_session = spark.SparkSession()
    database      = spark_session.query(start_time=start_time,end_time=end_time,fill=fill,variables = variables)
    #============================================

    # Adding proper timestamp
    #============================================
    database = database.sort_index()
    database.index.name = 'unix'
    database.insert(0,'Timestamp',database.index)
    database.insert(1,'Time',1e-9*(database.index - database.index[0]))
    database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(spark_session.TZONE))
    #============================================

    return database


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
def getFillingPattern(database,n_LR = 21):

    # Extracting pattern from intensity
    #=====================================================
    filled_b_slots = {}
    for beam in beams:
        
        # Only looking at after max of intensity
        bb_I    = database[beam['bb_Intensity']].dropna()
        max_loc = bb_I.apply(lambda line: np.sum(line)).argmax()
        bb_I    = bb_I.iloc[max_loc:]


        # Taking bb intensity just after max signal and then 1/4 of the way
        probe1 = bb_I.iloc[20-10:20+10].mean(axis=0)
        probe2 = bb_I.iloc[len(bb_I)//4-10:len(bb_I)//4+10].mean(axis=0)


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

    #=====================================================


    # # Adding train lengths
    # #=====================================================
    # for bb_patt in [bb_df_b1,bb_df_b2]:
    #     _lengths = get_train_lengths(bb_patt)
    #     train_idx     = np.repeat(np.arange(len(_lengths)),_lengths)
    #     train_lengths = np.repeat(_lengths,_lengths)

    #     bb_patt.insert(0,'Train',train_idx)
    #     bb_patt.insert(1,'Train Length',train_lengths)
    #     bb_patt.insert(2,'Tag',None)

    #     # Add Tag for bunch in trains:
    #     for name,train in bb_patt.groupby('Train'):
    #         length = train['Train Length'].unique()[0]   
    #         for i in range(length):
    #             _idx  = train.index[i]
    #             bb_patt.loc[_idx,'Tag']  = f'{i}/{length}'
    # #=====================================================

    return bb_df_b1,bb_df_b2
#=====================================================
