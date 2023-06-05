import numpy as np
import pandas as pd
from pathlib import Path
import gc

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



_default_out = '/eos/user/p/phbelang/Programming/dataBase/LuminosityFollowUp/2023/rawdata'

# To call the script directly
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("FILL", type=int,                              help = "Fill number to analyse")
    aparser.add_argument("-e", "--export",                              help = "Location to save data"    ,default = _default_out)
    args = aparser.parse_args()
    
    assert args.FILL>8000, 'Invalid fill number'
    assert Path(args.export).exists(), 'Invalid export path'
    
    print(40*'*')
    print(f' Downloading NXCALS data from FILL {args.FILL}\n | Export to: {args.export}/FILL{args.FILL}.parquet')
    print(40*'*')


    # Minimal variables to be used
    #============================================
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


    df = parser.from_NXCALS(fill=args.FILL,variables = variables)

    # Saving to parquet
    #============================================
    df.to_parquet(f'{args.export}/FILL{args.FILL}.parquet')