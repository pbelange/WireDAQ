
import numpy as np
import pandas as pd
import scipy.interpolate as sciInterp

import pynumdiff
import fillingpatterns as fp

import WireDAQ.NXCALS as nx
# Make sure this import is after pandas
import WireDAQ.PandasPlus



# Creating NXCALS variable containers
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
buckets   = np.arange(3564)


def importData(FILL,DATA_PATH = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2022_test_v0.2.2/'):
    # Everything except BPMs
    var_needed  = [_var for _var in beams[0]._getVarList() + beams[1]._getVarList()]
    var_needed += [wire.I for wire in wires['B1']+wires['B2']]
    var_needed += ['HX:BMODE','LHC.STATS:LHC:INJECTION_SCHEME']

    # Remove duplicates
    var_needed = list(set(var_needed))

    var_needed = [_var for _var in var_needed if ('BFC' not in _var)]
    var_needed = [_var for _var in var_needed if ('BQBB' not in _var)]
    var_needed.remove('ALICE:BUNCH_LUMI_INST')
    var_needed.remove('LHCB:BUNCH_LUMI_INST')

    tune_vars = ['LhcStateTracker:LHCBEAM1:QH_TRIM:value',
                'LhcStateTracker:LHCBEAM1:QV_TRIM:value',
                'LhcStateTracker:LHCBEAM2:QH_TRIM:value',
                'LhcStateTracker:LHCBEAM2:QV_TRIM:value']

    beta_vars = ['HX:BETASTAR_IP1',
                'HX:BETASTAR_IP2',
                'HX:BETASTAR_IP5',
                'HX:BETASTAR_IP8']

    # Importing fill data
    columns = var_needed+tune_vars + beta_vars
    columns.sort()
    database = pd.read_parquet(DATA_PATH+f'/HX:FILLN={FILL}',columns=columns)
    database = database.sort_index()
    database.index.name = 'unix'
    database.insert(0,'Timestamp',database.index)
    database.insert(1,'Time',1e-9*(database.index - database.index[0]))
    database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC'))

    return database

def getFillingPattern(jsonFile):

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

def computeLumiTot(df):

    # Starting with ATLAS
    df.insert(4,'bb_LUMI_TOT',df[beams[0]['bb_Luminosity']['ATLAS']])



    # Adding contribution from other IPs
    times,lumi_tot  = df['bb_LUMI_TOT'].to_2D()
    for loc in ['CMS']:
        times_found,allBunches = df[beams[0]['bb_Luminosity'][loc]].to_2D()

        lumi_tot   += sciInterp.interp1d(times_found,allBunches,axis=0,bounds_error=False,fill_value=np.nan)(times)

    df.loc[times,'bb_LUMI_TOT'] = pd.DataFrame({'tmp':list(lumi_tot)},index=times)['tmp']

    subset = df.dropna(subset='bb_LUMI_TOT')
    return subset[['Timestamp','Time','bb_LUMI_TOT']]


# Computing intensity avg (every 20 seconds)ç
def compute_BCT_avg(df,STABLE_start):

    # STABLE_start = df[df['HX:BMODE'] == 'STABLE']['Time'].iloc[0]

    # Computing intensity avg (every 20 seconds)
    binWind = 20

    _times   = df.set_index('Time')[beams[0]['bb_Luminosity']['ATLAS']].dropna().index
    bins    = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    new_dict = {}
    for beam in beams:

        observable = beam['Intensity']
        #observable = beam['Intensity'].replace('A6','B6')
        times,ref  = df.bin(observable,bins=bins)[['Time',observable]].T.to_numpy()
        STABLE_idx = np.argmin(np.abs(times-STABLE_start))

        observable = beam['bb_Intensity']
        total_A    = df.bin(observable,bins=bins)[observable]
        calib_A   = np.mean((ref/total_A.apply(lambda line:np.sum(line)))[STABLE_idx:STABLE_idx+int(10*(60/binWind))])

        new_dict[f'bb_Intensity_A_{beam.name}'] = total_A*calib_A

        observable = beam['bb_Intensity_B']
        total_B    = df.bin(observable,bins=bins)[observable]

        # B1 BCTFR-B needs rolling of 1 bucket (let's make it general):
        #-------------------------------------
        #if beam.name == 'B1':
        #    total_B = total_B.apply(lambda line: np.roll(line,1))
        
        rolling = buckets[(total_A.loc[STABLE_idx]!=0)&(buckets<3562)][-1]-buckets[(total_B.loc[STABLE_idx]!=0)&(buckets<3562)][-1]
        if rolling != 0 :
            total_B = total_B.apply(lambda line: np.roll(line,rolling))
            print(f'MISALIGNED {beam.name}, ROLLING BY {rolling}')
        
        #-------------------------------------

        
        calib_B   = np.mean((ref/total_B.apply(lambda line:np.sum(line)))[STABLE_idx:STABLE_idx+int(10*(60/binWind))])

        new_dict[f'bb_Intensity_B_{beam.name}']   = total_B*calib_B

        new_dict[f'bb_Intensity_avg_{beam.name}'] = (total_A*calib_A + total_B*calib_B)/2



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


def compute_lifetime_xsection(df,beam):

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps
    dt    = 60
    times = np.arange(tmp[0],tmp[-1],dt)
    bins  = np.diff(times)/2 + times[:-1]
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
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC') for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,'dNdt':list(dNdt),'Lifetime':list(lifetime),'sig_eff':list(xsection)},index=bin_unix)


def smooth_lifetime_xsection(df,beam):

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps
    dt    = 60
    times = np.arange(tmp[0],tmp[-1],dt)
    bins  = np.diff(times)/2 + times[:-1]
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
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC') for t in bin_unix]


    return pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,'dNdt':list(dNdt),'Lifetime':list(lifetime),'sig_eff':list(xsection)},index=bin_unix)
