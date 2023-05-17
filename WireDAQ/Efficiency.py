
import numpy as np
import pandas as pd
import scipy.interpolate as sciInterp
from pathlib import Path

import pynumdiff
# import fillingpatterns as fp

import WireDAQ.Constants as cst
import WireDAQ.NXCALS as nx
# Make sure this import is after pandas
import WireDAQ.PandasPlus



# Creating NXCALS variable containers
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
LHC       = nx.NXCALSLHC()
buckets   = np.arange(3564)

SIG_PP = cst.SIG_PP
TZONE  = cst.TZONE

def importFILL(FILL,DATA_PATH = '/home/lumimod/2022_07_26_NXCALS/2022_v3'):

    columns = [ 'LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:target',
                'LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:target',
                'HX:BMODE',
                'ATLAS:BUNCH_LUMI_INST',
                'ATLAS:LUMI_TOT_INST',
                'CMS:BUNCH_LUMI_INST',
                'CMS:LUMI_TOT_INST',
                'HX:BETASTAR_IP1',
                'HX:BETASTAR_IP5',
                'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',
                'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',
                'LHC.BCTDC.B6R4.B1:BEAM_INTENSITY',
                'LHC.BCTDC.B6R4.B2:BEAM_INTENSITY',
                'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY',
                'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY',
                'LHC.BCTFR.B6R4.B1:BUNCH_INTENSITY',
                'LHC.BCTFR.B6R4.B2:BUNCH_INTENSITY',
                'LHC.BQM.B1:NO_BUNCHES',
                'LHC.BQM.B2:NO_BUNCHES',
                'RPMC.UL14.RBBCW.L1B1:I_MEAS',
                'RPMC.UL16.RBBCW.R1B2:I_MEAS',
                'RPMC.UL557.RBBCW.R5B2:I_MEAS',
                'RPMC.USC55.RBBCW.L5B1:I_MEAS']


    database = pd.read_parquet(list(Path(DATA_PATH).glob(f'HX:FILLN={FILL}/HX:BMODE=STABLE/*.parquet')),columns=columns)
    database = database.sort_index()
    database.index.name = 'unix'
    database.insert(0,'Timestamp',database.index)
    database.insert(1,'Time',1e-9*(database.index - database.index[0]))
    database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE))

    return database

def import_MD_Data(parquetFile,allColumns = False):
    columns = [ 'LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:target',
                'LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:target',
                'HX:BMODE',
                'ATLAS:BUNCH_LUMI_INST',
                'ATLAS:LUMI_TOT_INST',
                'CMS:BUNCH_LUMI_INST',
                'CMS:LUMI_TOT_INST',
                'HX:BETASTAR_IP1',
                'HX:BETASTAR_IP2',
                'HX:BETASTAR_IP5',
                'HX:BETASTAR_IP8',
                'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',
                'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',
                'LHC.BCTDC.B6R4.B1:BEAM_INTENSITY',
                'LHC.BCTDC.B6R4.B2:BEAM_INTENSITY',
                'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY',
                'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY',
                'LHC.BCTFR.B6R4.B1:BUNCH_INTENSITY',
                'LHC.BCTFR.B6R4.B2:BUNCH_INTENSITY',
                'LHC.BQM.B1:NO_BUNCHES',
                'LHC.BQM.B2:NO_BUNCHES',
                'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H',
                'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V',
                'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H',
                'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V',
                'LhcStateTracker:LHCBEAM1:QH_TRIM:value',
                'LhcStateTracker:LHCBEAM1:QV_TRIM:value',
                'LhcStateTracker:LHCBEAM2:QH_TRIM:value',
                'LhcStateTracker:LHCBEAM2:QV_TRIM:value',
                'RPMC.UL14.RBBCW.L1B1:I_MEAS',
                'RPMC.UL16.RBBCW.R1B2:I_MEAS',
                'RPMC.UL557.RBBCW.R5B2:I_MEAS',
                'RPMC.USC55.RBBCW.L5B1:I_MEAS']

    if allColumns:
        database = pd.read_parquet(parquetFile)
    else:
        database = pd.read_parquet(parquetFile,columns=columns)
    database = database.sort_index()
    database.index.name = 'unix'
    database.insert(0,'Timestamp',database.index)
    database.insert(1,'Time',1e-9*(database.index - database.index[0]))
    database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE))

    return database

# importing dBLM
def import_dBLM_Data(parquetFile):
    # beams = [nx.NXCALSBeam(name) for name in ['B1','B2']]
    dblm_columns = [_var for _var in beams[0]._getVarList()+beams[1]._getVarList() if 'BLMD' in _var]

    database = pd.read_parquet(parquetFile,columns=dblm_columns)
    database = database.sort_index()
    database.index.name = 'unix'
    database.insert(0,'Timestamp',database.index)
    database.insert(1,'Time',1e-9*(database.index - database.index[0]))
    database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE))

    return database

    
def get_MD_FillingPattern(jsonFile):

    # Figuring out filling patterns
    patt = fp.FillingPattern.from_json(jsonFile) 
    patt.compute_beam_beam_schedule(n_lr_per_side=21)

    for bb_patt in [patt.b1,patt.b2]:
        train_num     = np.repeat(np.arange(len(bb_patt.train_lengths)),bb_patt.train_lengths)
        train_lengths = np.repeat(bb_patt.train_lengths,bb_patt.train_lengths)
        bb_patt.bb_schedule.insert(0,'Train',train_num)
        bb_patt.bb_schedule.insert(1,'Train Lengths',train_lengths)
        bb_patt.bb_schedule.insert(2,'Tag',None)


        targetTrain = bb_patt.bb_schedule['Train Lengths'].max()

        grouped = bb_patt.bb_schedule.groupby('Train')

        # Add Tag for bunch in trains:
        for train in list(grouped.groups.keys()):
            #if grouped.get_group(train)['Train Lengths'].max() == targetTrain:
            length = grouped.get_group(train)['Train Lengths'].max()   
            for i in range(length):
                _idx  = grouped.get_group(train).index[i]
                bb_patt.bb_schedule.loc[_idx,'Tag']  = f'{i}/{length}'

    return patt

def getFillingPattern(b1_bool,b2_bool):

    # Figuring out filling patterns
    patt = fp.FillingPattern(b1_bool,b2_bool) 
    patt.compute_beam_beam_schedule(n_lr_per_side=21)

    for bb_patt in [patt.b1,patt.b2]:
        train_num     = np.repeat(np.arange(len(bb_patt.train_lengths)),bb_patt.train_lengths)
        train_lengths = np.repeat(bb_patt.train_lengths,bb_patt.train_lengths)
        bb_patt.bb_schedule.insert(0,'Train',train_num)
        bb_patt.bb_schedule.insert(1,'Train Lengths',train_lengths)
        bb_patt.bb_schedule.insert(2,'Tag',None)


        targetTrain = bb_patt.bb_schedule['Train Lengths'].max()

        grouped = bb_patt.bb_schedule.groupby('Train')

        # Add Tag for bunch in trains:
        for train in list(grouped.groups.keys()):
            #if grouped.get_group(train)['Train Lengths'].max() == targetTrain:
            length = grouped.get_group(train)['Train Lengths'].max()   
            for i in range(length):
                _idx  = grouped.get_group(train).index[i]
                bb_patt.bb_schedule.loc[_idx,'Tag']  = f'{i}/{length}'

    return patt


def computeLumiTot(df):

    # Starting with ATLAS
    df.insert(4,'bb_LUMI_TOT',df[LHC['bb_Luminosity']['ATLAS']])


    # Adding contribution from other IPs
    times,lumi_tot  = df['bb_LUMI_TOT'].to_2D()
    for loc in ['CMS']:
        times_found,allBunches = df[LHC['bb_Luminosity'][loc]].to_2D()

        lumi_tot   += sciInterp.interp1d(times_found,allBunches,axis=0,bounds_error=False,fill_value=np.nan)(times)

    df.loc[times,'bb_LUMI_TOT'] = pd.DataFrame({'tmp':list(lumi_tot)},index=times)['tmp']

    subset = df.dropna(subset='bb_LUMI_TOT')
    return subset[['Timestamp','Time','bb_LUMI_TOT']]



def calibrate_BCT(df,calib_ts):

    # Computing intensity avg (every 20 seconds)
    binWind = 20

    _times   = df.set_index('Time')[LHC['bb_Luminosity']['ATLAS']].dropna().index
    bins    = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    calib_dict = {'B1':{'A':1.0,'B':1.0},'B2':{'A':1.0,'B':1.0}}
    for beam in beams:

        observable = beam['Intensity']
        
        times,ref  = df.bin(observable,bins=bins)[['Time',observable]].T.to_numpy()
        calib_time = df.at_ts(calib_ts,'Time',method='nearest')
        calib_idx  = np.argmin(np.abs(times-calib_time))

        observable = beam['bb_Intensity']
        total_A    = df.bin(observable,bins=bins)[observable]
        calib_A   = np.mean((ref/total_A.apply(lambda line:np.sum(line)))[calib_idx:calib_idx+int(10*(60/binWind))])

        observable = beam['bb_Intensity_B']
        total_B    = df.bin(observable,bins=bins)[observable]
        calib_B   = np.mean((ref/total_B.apply(lambda line:np.sum(line)))[calib_idx:calib_idx+int(10*(60/binWind))])

        calib_dict[beam.name]['A'] = calib_A
        calib_dict[beam.name]['B'] = calib_B

    return calib_dict



# Computing intensity avg (every 20 seconds)ç
def compute_BCT_avg(df,calib_ts):

    # Computing intensity avg (every 20 seconds)
    binWind = 20

    _times   = df.set_index('Time')[LHC['bb_Luminosity']['ATLAS']].dropna().index
    bins     = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    new_dict = {}
    for beam in beams:

        observable = beam['Intensity']
        
        times,ref  = df.bin(observable,bins=bins)[['Time',observable]].T.to_numpy()
        calib_time = df.at_ts(calib_ts,'Time',method='nearest')
        calib_idx  = np.argmin(np.abs(times-calib_time))


        observable = beam['bb_Intensity']
        total_A    = df.bin(observable,bins=bins)[observable]
        calib_A    = np.mean((ref/total_A.apply(lambda line:np.sum(line)))[calib_idx:calib_idx+int(10*(60/binWind))])

        new_dict[f'bb_Intensity_A_{beam.name}'] = total_A*calib_A

        observable = beam['bb_Intensity_B']
        total_B    = df.bin(observable,bins=bins)[observable]
        calib_B   = np.mean((ref/total_B.apply(lambda line:np.sum(line)))[calib_idx:calib_idx+int(10*(60/binWind))])

        # B1 BCTFR-B needs rolling of 1 bucket (let's make it general):
        #-------------------------------------
        
        rolling = buckets[(total_A.values[calib_idx]>0.5e10)&(buckets<3562)][-1]-buckets[(total_B.values[calib_idx]>0.5e10)&(buckets<3562)][-1]
        if rolling != 0 :
            total_B = total_B.apply(lambda line: np.roll(line,rolling))
            print(f'MISALIGNED {beam.name}, ROLLING BY {rolling}')
        
        #-------------------------------------

        new_dict[f'bb_Intensity_B_{beam.name}']   = total_B*calib_B

        new_dict[f'bb_Intensity_avg_{beam.name}'] = (new_dict[f'bb_Intensity_A_{beam.name}'] + new_dict[f'bb_Intensity_B_{beam.name}'])/2



    return pd.concat([  df.bin(beams[0]['Intensity'],bins=bins)[['unix','Timestamp','Time']],
                        pd.DataFrame(new_dict)],axis=1).set_index('unix')




def first_order_diff(x,dt):
    # Taken from
    # https://github.com/luckystarufo/PyNumDiff/blob/upgrade/pynumdiff/finite_difference/_finite_difference.py

    # Calculate the finite difference
    dxdt_hat = np.diff(x,axis=0) / dt
    
    # Pad the data
    dxdt_hat = np.vstack((dxdt_hat[0,:], dxdt_hat, dxdt_hat[-1,:]))
    
    # Re-finite dxdt_hat using linear interpolation
    dxdt_hat = np.mean((dxdt_hat[0:-1,:], dxdt_hat[1:,:]), axis=0)

    return dxdt_hat




def compute_BCTF_efficiency(df,beam):

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps and aligned at max of lumi
    dt      = 60
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])
    bins    = np.diff(times)/2 + times[:-1] 
    #-----------------------------------------

    #-----------------------------------------

    # Extracting data
    #-----------------------------------------
    observable     = f'bb_Intensity_avg_{beam.name}'
    times_N,data_N = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()

    observable     = 'bb_LUMI_TOT'
    times_L,data_L = df.set_index('Time')[observable].dropna().to_2D()
    #----------------------------------------


    # Aligning times with interpolation
    #---------------------------------
    times   = times[10:-2]
    data_N  =     sciInterp.interp1d(times_N,data_N,axis=0)(times)
    data_L  = 1e6*sciInterp.interp1d(times_L,data_L,axis=0)(times)
    #---------------------------------

    # Computing efficiency
    #---------------------------------
    dNdt     = first_order_diff(data_N,dt)
    eta      = SIG_PP*data_L/(-dNdt)
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity':list(data_L),
                            'Intensity' :list(data_N),
                            'dNdt'      :list(dNdt),
                            'eta'       :list(eta)},index=bin_unix)



def compute_dBLM_efficiency(df,beam,baseline = None,calib_ROI=None,filled_buckets=None):

    # Still binning every 10 seconds to reduce the noise
    binWind = 10

    _times   = df.set_index('Time')['Luminosity'].dropna().index
    bins     = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    # Extracting data
    #-----------------------------------------
    observable     = 'Luminosity'
    times_L,data_L = df.set_index('Time')[observable].dropna().to_2D()


    data_dblm = {}
    for dblmType in ['V','H-V-S']:
        observable       = beam.dBLM_Amp[dblmType]
        times_dblm,_data = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()
        data_dblm[dblmType] = _data
    #-----------------------------------------


    # Aligning times with interpolation
    #---------------------------------
    # IF NEEDED : bounds_error=False,fill_value=np.nan
    times     = times_dblm[10:-10]
    data_L    = sciInterp.interp1d(times_L,data_L,axis=0,bounds_error=False,fill_value=np.nan)(times)

    for dblmType in ['V','H-V-S']:
        data_dblm[dblmType]  = sciInterp.interp1d(times_dblm,data_dblm[dblmType],axis=0)(times)
    #---------------------------------


    # Computing xsection
    #==================================
    # Substracting baseline
    #---------------------------------
    if baseline is not None:
        assert(isinstance(baseline,dict)),'baseline must be specified for each bunch and each dblm type'
    else:
        baseline = {}
        for dblmType in ['V','H-V-S']:
            baseline[dblmType] = np.min(data_dblm[dblmType],axis=0)

    for dblmType in ['V','H-V-S']:
        data_dblm[dblmType] -= baseline[dblmType]
    #---------------------------------

    # Calibration:
    #---------------------------------
    if calib_ROI is not None:
        time_calib,data_calib = df.set_index('Timestamp')[calib_ROI[0]:calib_ROI[1]].set_index('Time')[f'Luminosity'].dropna().to_2D()
        lumi_losses           = np.trapz(80e-3*data_calib,time_calib,axis=0)

        time_calib,data_calib = df.set_index('Timestamp')[calib_ROI[0]:calib_ROI[1]].set_index('Time')[f'{beam.name}_dNdt'].dropna().to_2D()
        target_bb             = np.trapz(-data_calib,time_calib,axis=0)-lumi_losses
        # target_bb             = np.sum(np.trapz(-data_calib,time_calib,axis=0)-lumi_losses)

        tot_loss = {}
        _ROI = (times>time_calib[0])&(times<time_calib[-1])
        for dblmType in ['V','H-V-S']:
            tot_loss[dblmType] = np.trapz(data_dblm[dblmType][_ROI,:],times[_ROI],axis=0)
            # tot_loss[dblmType] = np.sum(np.trapz(data_dblm[dblmType][_ROI,:],times[_ROI],axis=0))

        calib_factor = target_bb /(tot_loss['V']+tot_loss['H-V-S'])
        for dblmType in ['V','H-V-S']:
            data_dblm[dblmType] *= np.mean(calib_factor[filled_buckets]) 
    #---------------------------------


    eta      = SIG_PP*data_L/(SIG_PP*data_L+data_dblm['V']+data_dblm['H-V-S'])
    #==================================

    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return    pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity'   :list(data_L),
                            'Losses V'     :list(data_dblm['V']),
                            'Losses H-V-S' :list(data_dblm['H-V-S']),
                            'eta'          :list(eta)},index=bin_unix)



def compute_loss_only_efficiency(df,beam,baseline = None,calib_ROI=None):

    # Still binning every 10 seconds to reduce the noise
    binWind = 10

    _times   = df.set_index('Time')[f'{beam.name}_dNdt'].dropna().index
    bins     = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    # Extracting data
    #-----------------------------------------
    observable     = f'{beam.name}_dNdt'
    times_dNdt,data_dNdt = df.set_index('Time')[observable].dropna().to_2D()


    data_dblm = {}
    for dblmType in ['V','H-V-S']:
        observable       = beam.dBLM_Amp[dblmType]
        times_dblm,_data = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()
        data_dblm[dblmType] = _data
    #-----------------------------------------


    # Aligning times with interpolation
    #---------------------------------
    # IF NEEDED : bounds_error=False,fill_value=np.nan
    times     = times_dblm[10:-10]
    data_dNdt    = sciInterp.interp1d(times_dNdt,data_dNdt,axis=0,bounds_error=False,fill_value=np.nan)(times)

    for dblmType in ['V','H-V-S']:
        data_dblm[dblmType]  = sciInterp.interp1d(times_dblm,data_dblm[dblmType],axis=0)(times)
    #---------------------------------


    # Computing xsection
    #==================================
    # Substracting baseline
    #---------------------------------
    if baseline is not None:
        assert(isinstance(baseline,dict)),'baseline must be specified for each bunch and each dblm type'
    else:
        baseline = {}
        for dblmType in ['V','H-V-S']:
            baseline[dblmType] = np.min(data_dblm[dblmType],axis=0)

    for dblmType in ['V','H-V-S']:
        data_dblm[dblmType] -= baseline[dblmType]
    #---------------------------------



    eta = (data_dNdt + (data_dblm['V']+data_dblm['H-V-S']))/(data_dNdt)
    #==================================

    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return    pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'dNdt'         :list(data_dNdt),
                            'Losses V'     :list(data_dblm['V']),
                            'Losses H-V-S' :list(data_dblm['H-V-S']),
                            'eta'          :list(eta)},index=bin_unix)


# ARCHIVES BELOW
#=============================================================================================================================
#=============================================================================================================================
#=============================================================================================================================

def compute_lifetime_xsection(df,beam):

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps and aligned at max of lumi
    dt      = 60
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])
    bins    = np.diff(times)/2 + times[:-1] 
    #-----------------------------------------

    #-----------------------------------------

    # Extracting data
    #-----------------------------------------
    observable     = f'bb_Intensity_avg_{beam.name}'
    times_N,data_N = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()

    observable     = 'bb_LUMI_TOT'
    times_L,data_L = df.set_index('Time')[observable].dropna().to_2D()
    #----------------------------------------


    # Aligning times with interpolation
    #---------------------------------
    times   = times[10:-2]
    data_N  =     sciInterp.interp1d(times_N,data_N,axis=0)(times)
    data_L  = 1e6*sciInterp.interp1d(times_L,data_L,axis=0)(times)
    #---------------------------------

    # Computing xsection and lifetime
    #---------------------------------
    dNdt     = first_order_diff(data_N,dt)
    xsection = -dNdt/data_L
    lifetime = -data_N/dNdt
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity':list(data_L),
                            'Intensity' :list(data_N),
                            'dNdt'      :list(dNdt),
                            'Lifetime'  :list(lifetime),
                            'sig_eff'   :list(xsection)},index=bin_unix)


def compute_total_lifetime_xsection(df,beam):

    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp     = df.set_index('Time')['bb_LUMI_TOT'].dropna().index


    # forcing constant time steps and aligned at max of lumi
    dt      = 60
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])
    bins    = np.diff(times)/2 + times[:-1] 

    #-----------------------------------------

    # Extracting data
    #-----------------------------------------
    observable = beam['Intensity']
    data_N     = df.bin(observable,bins=bins).set_index('Time')[observable].dropna()
    times_N     = data_N.index

    observable  = 'bb_LUMI_TOT'
    data_L      = df.set_index('Time')[observable].dropna().apply(lambda line: np.sum(line))
    times_L      = data_L.index
    #----------------------------------------


    # Aligning times with interpolation
    #---------------------------------
    times   = times[10:-2]
    data_N  =     sciInterp.interp1d(times_N,data_N,axis=0)(times)
    data_L  = 1e6*sciInterp.interp1d(times_L,data_L,axis=0)(times)
    #---------------------------------


    # plt.figure()
    # plt.plot(df.set_index('Time')[beam['Intensity']].dropna(),'-o',alpha=0.5)
    # plt.plot(times,data_N,'-o',ms=2,alpha=0.5)

    # plt.figure()
    # plt.plot(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)),'-o',alpha=0.5)
    # plt.plot(times,data_L/1e6,'-o',ms=2,alpha=0.5)

    # Computing xsection and lifetime
    #---------------------------------
    dNdt     = first_order_diff(np.array([data_N]).T,dt).flatten()
    xsection = -dNdt/data_L
    lifetime = -data_N/dNdt
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,'dNdt':list(dNdt),'Lifetime':list(lifetime),'sig_eff':list(xsection)},index=bin_unix)




def smooth_lifetime_xsection(df,beam):

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps and aligned at max of lumi
    dt      = 60
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])
    bins    = np.diff(times)/2 + times[:-1] 
    #-----------------------------------------

    # Extracting data
    #-----------------------------------------
    observable     = f'bb_Intensity_avg_{beam.name}'
    times_N,data_N = df.set_index('Time')[observable].dropna().to_2D()
    dt_N           = np.mean(np.diff(times_N))



    observable     = 'bb_LUMI_TOT'
    times_L,data_L = df.set_index('Time')[observable].dropna().to_2D()
    #----------------------------------------

    # Computing dNdt
    params = [18,2]
    dNdt   = np.nan*data_N.copy()
    for bucket in buckets:
        N_hat,dNdt_hat  = pynumdiff.smooth_finite_difference.meandiff(data_N[:,bucket], dt_N, params, options={'iterate': True})
        dNdt[:,bucket]   = dNdt_hat
        data_N[:,bucket] = N_hat 


    # Aligning times with interpolation
    #---------------------------------
    times   = times[10:-10]
    data_N  =     sciInterp.interp1d(times_N,data_N,axis=0)(times)
    dNdt    =     sciInterp.interp1d(times_N,dNdt  ,axis=0)(times)
    data_L  = 1e6*sciInterp.interp1d(times_L,data_L,axis=0)(times)
    #---------------------------------

    # Computing xsection and lifetime
    #---------------------------------
    xsection = -dNdt/data_L
    lifetime = -data_N/dNdt
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity':list(data_L),
                            'Intensity' :list(data_N),
                            'dNdt'      :list(dNdt),
                            'Lifetime'  :list(lifetime),
                            'sig_eff'   :list(xsection)},index=bin_unix)


def smooth_total_lifetime_xsection(df,beam):
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp     = df.set_index('Time')['bb_LUMI_TOT'].dropna().index


    # forcing constant time steps and aligned at max of lumi
    dt      = 60
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])
    bins    = np.diff(times)/2 + times[:-1] 

    #-----------------------------------------

    # Extracting data
    #-----------------------------------------
    observable = beam['Intensity']
    data_N     = df.bin(observable,bins=bins).set_index('Time')[observable].dropna()
    times_N     = data_N.index

    observable  = 'bb_LUMI_TOT'
    data_L      = df.set_index('Time')[observable].dropna().apply(lambda line: np.sum(line))
    times_L     = data_L.index
    #----------------------------------------



    # Computing dNdt
    params = [5,2]
    dt_N   = np.mean(np.diff(times_N))
    data_N = np.array([data_N]).T
    dNdt   = np.nan*data_N.copy()
    for bucket in [0]:
        N_hat,dNdt_hat  = pynumdiff.smooth_finite_difference.meandiff(data_N[:,bucket], dt_N, params, options={'iterate': True})
        dNdt[:,bucket]   = dNdt_hat
        data_N[:,bucket] = N_hat 

    data_N = data_N.flatten()
    dNdt   = dNdt.flatten()


    # Aligning times with interpolation
    #---------------------------------
    times   = times[10:-10]
    data_N  =     sciInterp.interp1d(times_N,data_N,axis=0)(times)
    dNdt    =     sciInterp.interp1d(times_N,dNdt  ,axis=0)(times)
    data_L  = 1e6*sciInterp.interp1d(times_L,data_L,axis=0)(times)
    #---------------------------------

    # Computing xsection and lifetime
    #---------------------------------
    xsection = -dNdt/data_L
    lifetime = -data_N/dNdt
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,'dNdt':list(dNdt),'Lifetime':list(lifetime),'sig_eff':list(xsection)},index=bin_unix)


def compute_dBLM_xsection(df,beam,baseline = None,calib_ROI=None):

    # Still binning every 10 seconds to reduce the noise
    binWind = 10

    _times   = df.set_index('Time')['Luminosity'].dropna().index
    bins     = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    # Extracting data
    #-----------------------------------------
    observable     = 'Luminosity'
    times_L,data_L = df.set_index('Time')[observable].dropna().to_2D()


    data_dblm = {}
    for dblmType in ['V','H-V-S']:
        observable       = beam.dBLM_Amp[dblmType]
        times_dblm,_data = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()
        data_dblm[dblmType] = _data
    #-----------------------------------------


    # Aligning times with interpolation
    #---------------------------------
    # IF NEEDED : bounds_error=False,fill_value=np.nan
    times     = times_dblm[10:-10]
    data_L    = sciInterp.interp1d(times_L,data_L,axis=0,bounds_error=False,fill_value=np.nan)(times)

    for dblmType in ['V','H-V-S']:
        data_dblm[dblmType]  = sciInterp.interp1d(times_dblm,data_dblm[dblmType],axis=0)(times)
    #---------------------------------


    # Computing xsection
    #==================================
    # Substracting baseline
    #---------------------------------
    if baseline is not None:
        assert(isinstance(baseline,dict)),'baseline must be specified for each bunch and each dblm type'
    else:
        baseline = {}
        for dblmType in ['V','H-V-S']:
            baseline[dblmType] = np.min(data_dblm[dblmType],axis=0)

    for dblmType in ['V','H-V-S']:
        data_dblm[dblmType] -= baseline[dblmType]
    #---------------------------------

    # Calibration:
    # #---------------------------------
    # time_calib,data_calib = dblm.set_index('Timestamp')[calib_ROI[0]:calib_ROI[1]].set_index('Time')[f'Luminosity'].dropna().to_2D()
    # lumi_losses           = np.trapz(80e-3*data_calib,time_calib,axis=0)

    # time_calib,data_calib = dblm.set_index('Timestamp')[calib_ROI[0]:calib_ROI[1]].set_index('Time')[f'{beam.name}_dNdt'].dropna().to_2D()
    # target_bb             = np.trapz(-data_calib,time_calib,axis=0)-lumi_losses
    # # target_bb             = np.sum(np.trapz(-data_calib,time_calib,axis=0)-lumi_losses)

    # tot_loss = {}
    # _ROI = (times>time_calib[0])&(times<time_calib[-1])
    # for dblmType in ['V','H-V-S']:
    #     tot_loss[dblmType] = np.trapz(data_dblm[dblmType][_ROI,:],times[_ROI],axis=0)
    #     # tot_loss[dblmType] = np.sum(np.trapz(data_dblm[dblmType][_ROI,:],times[_ROI],axis=0))

    # calib_factor = target_bb /(tot_loss['V']+tot_loss['H-V-S'])
    # for dblmType in ['V','H-V-S']:
    #     data_dblm[dblmType] *= calib_factor 
    # #---------------------------------

    xsection = 80e-3 + (data_dblm['V']+data_dblm['H-V-S'])/data_L
    #==================================

    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


    return    pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity'   :list(data_L),
                            'Losses V'     :list(data_dblm['V']),
                            'Losses H-V-S' :list(data_dblm['H-V-S']),
                            'sig_eff'      :list(xsection)},index=bin_unix)