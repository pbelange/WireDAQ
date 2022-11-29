import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import pandas as pd
import glob
import yaml
from pathlib import Path

import sys
sys.path.append('../')

import WireDAQ.PandasPlus

import WireDAQ.Constants as cst
import WireDAQ.NXCALS as nx
import WireDAQ.SigmaEff as sEff

import fillingpatterns as fp


# Creating NXCALS variable containers
#============================
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
LHC       = nx.NXCALSLHC()
buckets   = np.arange(3564)
#============================


TZONE = 'Europe/Paris'

# Variables for BCTF:
# ['HX:BMODE',
#  'HX:BETASTAR_IP1',
#  'HX:BETASTAR_IP5',
#  'ATLAS:BUNCH_LUMI_INST',
#  'CMS:BUNCH_LUMI_INST',
#  'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY',
#  'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY',
#  'LHC.BCTFR.B6R4.B1:BUNCH_INTENSITY',
#  'LHC.BCTFR.B6R4.B2:BUNCH_INTENSITY',
#  'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',
#  'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',
#  'LHC.BCTDC.B6R4.B1:BEAM_INTENSITY',
#  'LHC.BCTDC.B6R4.B2:BEAM_INTENSITY']

# Variables for dblm:
# ['HC.TZ76.BLMDIAMOND2.3:AcquisitionIntegral:intSumBuf0',
#  'HC.TZ76.BLMDIAMOND3.3:AcquisitionIntegral:intSumBuf0',
#  'HC.TZ76.BLMDIAMOND2.3:AcquisitionIntegral:intSumBuf1',
#  'HC.TZ76.BLMDIAMOND3.3:AcquisitionIntegral:intSumBuf1',
#  'HC.TZ76.BLMDIAMOND2.5:AcquisitionIntegral:intSumBuf0',
#  'HC.TZ76.BLMDIAMOND3.5:AcquisitionIntegral:intSumBuf0',
#  'HC.TZ76.BLMDIAMOND2.5:AcquisitionIntegral:intSumBuf1',
#  'HC.TZ76.BLMDIAMOND3.5:AcquisitionIntegral:intSumBuf1']


def compute_all_xsection(parquetfile_BCTF,parquetfile_dblm):
    var_needed = [  'HX:BMODE',
                    LHC['betastar']['IP1'],
                    LHC['betastar']['IP5'],
                    LHC['bb_Luminosity']['ATLAS'],
                    LHC['bb_Luminosity']['CMS']]

    var_needed += [beam['bb_Intensity'] for beam in beams]
    var_needed += [beam['bb_Intensity_B'] for beam in beams]


    var_needed += [beam['Intensity'] for beam in beams]
    var_needed += [beam['Intensity_B'] for beam in beams]
    



    # Importing data
    database = sEff.parquet2df(parquetfile_BCTF,columns=var_needed)

    # Croping data to last if needed
    MD_start = pd.Timestamp('2022-11-23 17:30:00',tz=TZONE)
    MD_stop  = pd.Timestamp('2022-11-23 19:45:00',tz=TZONE)

    database = database[(MD_start<database['Timestamp'])&
                        (MD_stop>database['Timestamp'])]
    #============================


    # Analysis
    #============================
    beta_start = database[database[LHC.betastar['IP1']]==60]['Timestamp'].min()

    # Calibrating when betastar reaches its final value
    #===============================
    calib_ts = beta_start
    BCT_avg  = sEff.compute_BCT_avg(database,calib_ts = calib_ts)
    database = pd.concat([database,BCT_avg])
    database = database.sort_index()

    # # Computing Lumi tot (ATLAS + CMS bunch by bunch)
    #===============================
    Lumi_tot = sEff.computeLumiTot(database)
    database = pd.concat([database,Lumi_tot])
    database = database.sort_index()


    # Figuring out filling patterns
    #===============================
    patterns       = {}
    for beam in beams:
        bunch_I  = database.set_index('Timestamp')[beam['bb_Intensity']].dropna().at_idx(calib_ts,method='nearest')
        patterns[beam.name] = (bunch_I>0.5e10)

    patt = sEff.getFillingPattern(patterns['B1'], patterns['B2'])
    #===============================



    # Computing effective cross section,
    xsection_BCTF = {}
    for beam in beams:
        xsection_BCTF[beam.name] = sEff.compute_lifetime_xsection(database,beam)


    # Importing dblm data
    #===============================
    var_needed = [_var for _var in beams[0]._getVarList()+beams[1]._getVarList() if 'BLMD' in _var]
    dblm = sEff.parquet2df(parquetfile_dblm,columns=var_needed)  

    dblm = dblm[(MD_start<dblm['Timestamp'])&
                (MD_stop>dblm['Timestamp'])]

    # Adding Lumi and losses info
    for beam in beams:
        toAdd = xsection_BCTF[beam.name][['Timestamp','Time','Luminosity','dNdt']].rename(columns={'dNdt':f'{beam.name}_dNdt'})
        toAdd.index.name = 'unix'
        if beam.name == 'B1':
            dblm = pd.concat([dblm,toAdd])
        else:
            dblm = pd.concat([dblm,toAdd.drop(columns=['Luminosity'])])
    dblm = dblm.sort_index()
    #============================


    # Computing effective cross section
    xsection_dblm = {}
    for beam in beams:
        xsection_dblm[beam.name] = sEff.compute_dBLM_xsection(dblm,beam,baseline = None,calib_ROI=None)



    return patt,xsection_BCTF,xsection_dblm






patt,xsection_BCTF,xsection_dblm = compute_all_xsection(parquetfile_BCTF ='../MD7003/Fullparquet/fullMD7003.parquet',
                                                        parquetfile_dblm ='../MD7003/Fullparquet/dBLM_MD7003.parquet')


