
#%%
FILL      = 8146
FIGWIDTH  = 12

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import pandas as pd
import glob
import sys
sys.path.append('../')

# Make sure this import is after pandas
import Backend.ABPPandas

import Backend.Constants as cst
import Backend.NXCALS as nx


# Matplotlib config
for key in plt.rcParams.keys():
    if 'date.auto' in key:
        plt.rcParams[key] = "%H:%M"


# Creating NXCALS variable containers
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]

buckets   = np.arange(3564)

# Locating data
DATA_PATH = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2022_test_v0.2.2/'
BMODE     = None#'STABLE'

# Importing fill data
database = pd.read_parquet(DATA_PATH+f'HX:FILLN={FILL}')
database = database.sort_index()
database.index.name = 'unix'
database.insert(0,'Timestamp',database.index)
database.insert(1,'Time',1e-9*(database.index - database.index[0]))
database['Timestamp'] = database['Timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC'))



from scipy.interpolate import interp1d

# COMPUTING LHCB BB-LUMI:
injection_scheme = database['LHC.STATS:LHC:INJECTION_SCHEME'].iloc[0]
nb_coll = {'ATLAS':int(injection_scheme.split('_')[2]),'CMS':int(injection_scheme.split('_')[2]),'LHCB':int(injection_scheme.split('_')[4]),'ALICE':int(injection_scheme.split('_')[3])}

nb_coll = {'ATLAS':len(buckets),'CMS':len(buckets),'LHCB':len(buckets),'ALICE':len(buckets)}

try:
    for loc in ['ALICE','LHCB']:    
        database.insert(5,f'{loc}:BUNCH_LUMI_INST',database[beams[0]['bb_Luminosity']['ATLAS']])

        times         = database[beams[0]['bb_Luminosity']['ATLAS']].dropna().index
        interp_values = np.interp(times, database[f'{loc}:LUMI_TOT_INST'].dropna().index ,database[f'{loc}:LUMI_TOT_INST'].dropna()/nb_coll[loc])
        database.loc[times,f'{loc}:BUNCH_LUMI_INST'] = database[beams[0]['bb_Luminosity']['ATLAS']].dropna()*0 + interp_values
except:
    pass

# COMPUTING LUMI TOT
#============================
try:
    database.insert(4,'bb_LUMI_TOT',database[beams[0]['bb_Luminosity']['ATLAS']])
except:
    database['bb_LUMI_TOT'] = database[beams[0]['bb_Luminosity']['ATLAS']].copy()


times,lumi_tot  = database['bb_LUMI_TOT'].to_2D()
for loc in ['ALICE','CMS','LHCB']:
    times_found,allBunches = database[beams[0]['bb_Luminosity'][loc]].to_2D()

    lumi_tot   += interp1d(times_found,allBunches,axis=0)(times)

database.loc[times,'bb_LUMI_TOT'] = pd.DataFrame({'tmp':list(lumi_tot)},index=times)['tmp']

#============================


def compute_lifetime_avg(df,beam):

    
    # Extracting quantities:
    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    times = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    bins  = np.diff(times)/2 + times[:-1]

    observable     = beam['bb_Intensity']
    times_N,data_N = df.bin(observable,bins=bins).set_index('Time')[observable].to_2D()

    observable  = 'bb_LUMI_TOT'
    _,data_L    = df.set_index('Time')[observable].to_2D()
    data_L      = 1e6*data_L[2:-2,:]
    times       = times[2:-2]
    #----------------------------------------

    # Aligning times with interpolation
    #---------------------------------
    data_N = interp1d(times_N,data_N,axis=0)(times)
    #---------------------------------

    # Computing xsection and lifetime
    #---------------------------------
    dNdt      = np.diff(data_N,axis=0)/np.tile(np.diff(times),(np.shape(data_N)[1],1)).T
    #xsection  = -dNdt/data_L[:-1,:]
    #N0        = np.tile(data_N[0,:],(np.shape(data_N)[0],1))
    lifetime  = -data_N[:-1,:]/dNdt
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC') for t in bin_unix]


    return pd.DataFrame({'Time':times[:-1],'Timestamp':bin_timestamp[:-1],'Lifetime':list(lifetime)},index=bin_unix[:-1])



def compute_xsection_avg(df,beam):

    
    # Extracting quantities:
    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    times = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    bins  = np.diff(times)/2 + times[:-1]

    observable     = beam['bb_Intensity']
    times_N,data_N = df.bin(observable,bins=bins).set_index('Time')[observable].to_2D()

    observable  = 'bb_LUMI_TOT'
    _,data_L    = df.set_index('Time')[observable].to_2D()
    data_L      = 1e6*data_L[2:-2,:]
    times       = times[2:-2]
    #----------------------------------------

    # Aligning times with interpolation
    #---------------------------------
    data_N = interp1d(times_N,data_N,axis=0)(times)
    #---------------------------------

    # Computing xsection and lifetime
    #---------------------------------
    dNdt      = np.diff(data_N,axis=0)/np.tile(np.diff(times),(np.shape(data_N)[1],1)).T
    xsection  = -dNdt/data_L[:-1,:]
    
    
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC') for t in bin_unix]


    return pd.DataFrame({'Time':times[:-1],'Timestamp':bin_timestamp[:-1],'sig_c':list(xsection)},index=bin_unix[:-1])






vmin = 80
vmax = 250
cmap = 'magma'

start_ts = pd.Timestamp('2022-08-20 10:00').tz_localize('UTC')
stop_ts  = pd.Timestamp('2022-08-20 12:00').tz_localize('UTC')

df = database
wire_start = df[df[wires['B1'][0]['I']]>10]['Timestamp'].min()
wire_stop  = df[(df[wires['B1'][0]['I']]>295)&(df['Timestamp']>wire_start)]['Timestamp'].min()


for beam in beams:
    fig = plt.figure(figsize=(12,6))
    plt.suptitle(f'{beam.name}, FILL {FILL} @ {str(start_ts)[:19]}')

    #computing filled buckets
    #-----------------------------
    filled_ts = np.min(database[database[beam['Intensity']]== np.max(database[beam['Intensity']])].index)

    sub = database.dropna(subset=[beam['bb_Intensity']])
    _ts = sub.index[sub.index.get_loc(filled_ts, method='nearest')]
    bunch_I   = database.loc[_ts,beam['bb_Intensity']]
    
    filled_buckets = np.arange(len(buckets))[bunch_I!=0]
    #-----------------------------


    _df = compute_xsection_avg(database,beam)
    times,datagrid = _df.set_index('Timestamp')['sig_c'].to_2D()

    #datagrid = datagrid[:,filled_buckets]
    
    plt.pcolormesh(times, np.arange(np.shape(datagrid)[1]), datagrid.T/1e-3,vmin=vmin, vmax=vmax,cmap=cmap)
    plt.axvline(wire_start-pd.Timedelta(minutes=3),color='lime')
    plt.axvline(wire_stop+pd.Timedelta(minutes=3),color='lime')

    cbar = plt.colorbar()
    cbar.set_label(r'$\sigma_{eff}$ (mb)')
    plt.ylabel('Bunch Buckets')
    plt.xlim([start_ts,stop_ts])

    plt.tight_layout()


for beam in beams:
    fig = plt.figure(figsize=(12,6))
    plt.suptitle(f'{beam.name}, FILL {FILL} @ {str(start_ts)[:19]}')
    
    #computing filled buckets
    #-----------------------------
    filled_ts = np.min(database[database[beam['Intensity']]== np.max(database[beam['Intensity']])].index)

    sub = database.dropna(subset=[beam['bb_Intensity']])
    _ts = sub.index[sub.index.get_loc(filled_ts, method='nearest')]
    bunch_I   = database.loc[_ts,beam['bb_Intensity']]
    
    filled_buckets = np.arange(len(buckets))[bunch_I!=0]
    #-----------------------------


    _df = compute_xsection_avg(database,beam)
    times,datagrid = _df.set_index('Timestamp')['sig_c'].to_2D()

    datagrid = datagrid[:,filled_buckets]
    
    plt.pcolormesh(times, np.arange(np.shape(datagrid)[1]), datagrid.T/1e-3,vmin=vmin,vmax=vmax,cmap=cmap)
    plt.axvline(wire_start-pd.Timedelta(minutes=3),color='lime')
    plt.axvline(wire_stop+pd.Timedelta(minutes=3),color='lime')

    cbar = plt.colorbar()
    cbar.set_label(r'$\sigma_{eff}$ (mb)')
    plt.ylabel('Filled Buckets')
    #plt.xlim([start_ts,stop_ts])
    plt.xlim([start_ts,pd.Timestamp('2022-08-20 12:00').tz_localize('UTC')])

    plt.tight_layout()



# %%

df = database
wire_start = df[df[wires['B1'][0]['I']]>10]['Timestamp'].min()
wire_stop  = df[(df[wires['B1'][0]['I']]>295)&(df['Timestamp']>wire_start)]['Timestamp'].min()


for _IP in ['IP1','IP5']:
    fig = plt.figure(figsize=(FIGWIDTH,2))
    for wire,color in zip(wires['B1']+wires['B2'],['C0','C0','C3','C3']):
        if wire.loc[1] == _IP[-1]:
            database.nxPlot('Timestamp',wire['I'],'.',alpha=0.3,color=color,label=wire.loc)
    plt.ylim([-5,375])
    plt.yticks([0,175,350])
    plt.ylabel('I (A)')
    plt.legend()
    plt.axvline(wire_start-pd.Timedelta(minutes=3),color='lime')
    plt.axvline(wire_stop+pd.Timedelta(minutes=3),color='lime')
    plt.xlim([start_ts,stop_ts])

# %%


plt.figure(figsize=(17,7))
for ii in ['LhcStateTracker:LHCBEAM1:QH_TRIM:target',
           'LhcStateTracker:LHCBEAM1:QV_TRIM:target',
           'LhcStateTracker:LHCBEAM2:QH_TRIM:target',
           'LhcStateTracker:LHCBEAM2:QV_TRIM:target']:
    

    time = database.set_index('Timestamp')[ii].dropna().index[0:-1:1000]
    data = database[ii].dropna().iloc[0:-1:1000]

    #database.nxPlot('Timestamp',ii,'.-',alpha=0.3,label=ii)
    plt.plot(time,data,'.-',alpha=0.3,label=ii)
    #plt.plot(at.pandas_index_localize(data_df[ii].dropna().iloc[0:-1:1000]),'.-',label=ii)
plt.axvline(wire_start-pd.Timedelta(minutes=3),color='lime')
plt.axvline(wire_stop+pd.Timedelta(minutes=3),color='lime')

plt.xlim([start_ts,stop_ts])

plt.legend(loc='best')
# %%
