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



_default_path   = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/'
# _default_path   = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2022/'
_default_out    = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/efficiency_data/BCTF'




def make_eff_df(FILL,data_path=_default_path):

    # Fixing data path
    raw_data    = data_path + '/rawdata/'

    # Declaring master bin times
    #-------------------------------------------------
    dt = 20
    unix_s,unix_e = parser.fill_unix_times(FILL,data_path=raw_data )
    unix_bins     = np.arange(unix_s,unix_e,dt/1e-9)
    #-------------------------------------------------

    # Import and bin dBLM 
    #-------------------------------------------------
    variables = []
    for beam in beams:
        variables += beam._getVarList(subset=['bb_Intensity','bb_Intensity_B'])

    bctf = parser.from_parquet2bin(fill=FILL,variables = variables,bins=unix_bins,beamMode = None,data_path= raw_data )
    bct  = parser.from_parquet2bin(fill=FILL,variables = [beam.Intensity for beam in beams],bins=unix_bins,keeptype=False,data_path= raw_data )

    _tmp    = pd.concat([bct,bctf],axis=1).sort_index()
    bct_avg = eff.compute_BCT_avg(_tmp)
    #-------------------------------------------------


    # Import lumi and compute total lumi
    #-------------------------------------------------
    variables  = [LHC.bb_Luminosity['ATLAS'],LHC.bb_Luminosity['CMS']]
    df_lumi     = parser.from_parquet(fill=FILL,variables = variables,beamMode = None,data_path= raw_data )
    df_lumi_tot = eff.compute_lumi_tot(df_lumi,experiments = ['ATLAS','CMS'])
    #-------------------------------------------------

    # Computing Efficiency
    #------------------------------------------------- 
    _tmp   = pd.concat([bct_avg,df_lumi_tot],axis=1).sort_index()
    df_eff = eff.compute_BCTF_efficiency(_tmp,beams, smooth_window= 9 ,smooth_iter = 1)


    return df_eff








# To call the script directly
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("FILL", type=int,      help = "Fill number to analyse")
    aparser.add_argument("-p", "--path",        help = "Location of parquet files"  ,default = _default_path)
    aparser.add_argument("-e", "--export",      help = "Location to export efficiency parquet"    ,default = _default_out)
    args = aparser.parse_args()
    
    assert args.FILL>8000, 'Invalid fill number'
    assert Path(args.path).exists(), 'Invalid data path'
    assert Path(args.export).exists(), 'Invalid export path'
    
    print(40*'*')
    print(f' dBLM_efficiency on F{args.FILL}\n | Data path: \t{args.path}\n | Parquet to: \t{args.export}/FILL{args.FILL}.parquet')
    print(40*'*')

    #===========================
    print('\n'+40*'=')
    print('Computing efficiency')
    print(40*'-')
    df_eff = make_eff_df(args.FILL,data_path=args.path)
    print('Saving to parquet')
    df_eff.to_parquet(f'{args.export}/FILL{args.FILL}.parquet')
    print(40*'=')

    gc.collect()
    #===========================


