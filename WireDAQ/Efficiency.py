
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.interpolate as sciInterp
import pynumdiff

# Local imports
#------------------------------------------------
import WireDAQ.PandasPlus           # Make sure this import is after pandas
import WireDAQ.Constants as cst
import WireDAQ.NXCALS as nx
#------------------------------------------------


# Creating NXCALS variable containers
#------------------------------------------------
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
LHC       = nx.NXCALSLHC()
b_slots   = np.arange(3564)
#------------------------------------------------




#================================================
# Utilities
#================================================
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



def computeLumiTot(df,experiments = ['ATLAS','CMS']):

    # Starting with first detector
    df.insert(4,'bb_LUMI_TOT',df[LHC['bb_Luminosity'][experiments[0]]])


    # Adding contribution from other IPs
    times,lumi_tot  = df['bb_LUMI_TOT'].to_2D()
    for loc in experiments[1:]:
        times_found,allBunches = df[LHC['bb_Luminosity'][loc]].to_2D()
        lumi_tot   += sciInterp.interp1d(times_found,allBunches,axis=0,bounds_error=False,fill_value=np.nan)(times)

    df.loc[times,'bb_LUMI_TOT'] = pd.DataFrame({'tmp':list(lumi_tot)},index=times)['tmp']

    subset = df.dropna(subset='bb_LUMI_TOT')
    return subset[['Timestamp','Time','bb_LUMI_TOT']]


def calibrate_BCT(df,dt=20,calibration_ts = None,reference = LHC['bb_Luminosity']['ATLAS']):

    # Computing intensity avg (every 20 seconds)
    binWind = dt

    # Calibration around max intensity
    if calibration_ts is None:
        _I      = df.set_index('Timestamp')[beams[0]['Intensity']].dropna()
        max_loc = _I.argmax()
        calibration_ts = _I.index[max_loc+20]

    _times   = df.set_index('Time')[reference].dropna().index
    bins     = np.arange(_times[0]+binWind/2,_times[-1],binWind)

    new_dict = {}
    for beam in beams:

        observable = beam['Intensity']
        
        times,ref  = df.bin(observable,bins=bins)[['Time',observable]].T.to_numpy()
        calib_time = df.at_ts(calibration_ts,'Time',method='nearest')
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
        
        rolling = b_slots[(total_A.values[calib_idx]>0.5e10)&(b_slots<3562)][-1]-b_slots[(total_B.values[calib_idx]>0.5e10)&(b_slots<3562)][-1]
        if rolling != 0 :
            total_B = total_B.apply(lambda line: np.roll(line,rolling))
            print(f'MISALIGNED {beam.name}, ROLLING BY {rolling}')
        
        #-------------------------------------

        new_dict[f'bb_Intensity_B_{beam.name}']   = total_B*calib_B

        new_dict[f'bb_Intensity_avg_{beam.name}'] = (new_dict[f'bb_Intensity_A_{beam.name}'] + new_dict[f'bb_Intensity_B_{beam.name}'])/2



    return pd.concat([  df.bin(beams[0]['Intensity'],bins=bins)[['unix','Timestamp','Time']],
                        pd.DataFrame(new_dict)],axis=1).set_index('unix')

#================================================


#============================================================================
# Efficiency calculations
#============================================================================

# FAST BCT
#----------------------------------
def compute_BCTF_efficiency(df,beam,dt= 60,smooth_derivative=False):

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps and aligned at max of lumi
    dt      = dt
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])

    assert dt > np.mean(np.diff(tmp)),f'dt must be larger than the sampling rate of the luminosity: {np.mean(np.diff(tmp))} s'
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
    if smooth_derivative:
        params = [18,2]
        dNdt   = np.nan*data_N.copy()
        for bucket in b_slots:
            N_hat,dNdt_hat   = pynumdiff.smooth_finite_difference.meandiff(data_N[:,bucket], np.mean(np.diff(times_N)), params, options={'iterate': True})
            dNdt[:,bucket]   = dNdt_hat
            data_N[:,bucket] = N_hat

        dNdt = sciInterp.interp1d(times_N,dNdt,axis=0)(times) 
    else:   
        dNdt = first_order_diff(data_N,dt)

    eta      = cst.SIG_PP*data_L/(-dNdt)
    sig_eff  = cst.SIG_PP/eta
    lifetime = -data_N/(dNdt-cst.SIG_PP*data_L)
    #---------------------------------


    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(cst.TZONE) for t in bin_unix]


    return pd.DataFrame({   'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity':list(data_L),
                            'Intensity' :list(data_N),
                            'dNdt'      :list(dNdt),
                            'eta'       :list(eta),
                            'sig_eff'   :list(sig_eff),
                            'lifetime'  :list(lifetime)},index=bin_unix)



# Diamond BLM
#----------------------------------
def compute_dBLM_efficiency(df,beam,dt = 10,baseline = None,calib_ROI=None,filled_b_slots=None):

    # Still binning every 10 seconds to reduce the noise

    #----------------------------------------
    # Luminosity is measured every minute, let's use it without too much manipulations
    tmp = df.set_index('Time')['bb_LUMI_TOT'].dropna().index
    
    # forcing constant time steps and aligned at max of lumi
    dt      = dt
    idx_max = np.argmax(df.set_index('Time')['bb_LUMI_TOT'].dropna().apply(lambda line: np.sum(line)))
    _times   = np.concatenate([np.arange(tmp[idx_max]-dt,tmp[0],-dt)[::-1],np.arange(tmp[idx_max],tmp[-1],dt)])


    # assert dt > np.mean(np.diff(_times)),f'dt must be larger than the sampling rate of the luminosity: {np.mean(np.diff(_times))} s'
    bins     = np.arange(_times[0]+dt/2,_times[-1],dt)

    # Extracting data
    #-----------------------------------------
    observable     = f'bb_Intensity_avg_{beam.name}'
    times_N,data_N = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()

    observable     = 'bb_LUMI_TOT'
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
    data_N    = sciInterp.interp1d(times_N,data_N,axis=0,bounds_error=False,fill_value=np.nan)(times)
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
            data_dblm[dblmType] *= np.mean(calib_factor[filled_b_slots]) 
    #---------------------------------


    # Computing efficiency
    #=================================
    R_ellN   = data_dblm['V']+data_dblm['H-V-S']
    dNdt     = cst.SIG_PP*data_L+R_ellN
    eta      = cst.SIG_PP*data_L/(cst.SIG_PP*data_L+R_ellN)
    sig_eff  = cst.SIG_PP/eta
    lifetime = -data_N/(R_ellN)
    #==================================

    # UNPACKING RESULTS
    bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(cst.TZONE) for t in bin_unix]


    return    pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
                            'Luminosity'   :list(data_L),
                            'Intensity'    :list(data_N),
                            'dNdt'         :list(dNdt),
                            'Losses V'     :list(data_dblm['V']),
                            'Losses H-V-S' :list(data_dblm['H-V-S']),
                            'eta'          :list(eta),
                            'sig_eff'      :list(sig_eff),
                            'lifetime'     :list(lifetime)},index=bin_unix)



# def compute_loss_only_efficiency(df,beam,baseline = None,calib_ROI=None):

#     # Still binning every 10 seconds to reduce the noise
#     binWind = 10

#     _times   = df.set_index('Time')[f'{beam.name}_dNdt'].dropna().index
#     bins     = np.arange(_times[0]+binWind/2,_times[-1],binWind)

#     # Extracting data
#     #-----------------------------------------
#     observable     = f'{beam.name}_dNdt'
#     times_dNdt,data_dNdt = df.set_index('Time')[observable].dropna().to_2D()


#     data_dblm = {}
#     for dblmType in ['V','H-V-S']:
#         observable       = beam.dBLM_Amp[dblmType]
#         times_dblm,_data = df.bin(observable,bins=bins).set_index('Time')[observable].dropna().to_2D()
#         data_dblm[dblmType] = _data
#     #-----------------------------------------


#     # Aligning times with interpolation
#     #---------------------------------
#     # IF NEEDED : bounds_error=False,fill_value=np.nan
#     times     = times_dblm[10:-10]
#     data_dNdt    = sciInterp.interp1d(times_dNdt,data_dNdt,axis=0,bounds_error=False,fill_value=np.nan)(times)

#     for dblmType in ['V','H-V-S']:
#         data_dblm[dblmType]  = sciInterp.interp1d(times_dblm,data_dblm[dblmType],axis=0)(times)
#     #---------------------------------


#     # Computing xsection
#     #==================================
#     # Substracting baseline
#     #---------------------------------
#     if baseline is not None:
#         assert(isinstance(baseline,dict)),'baseline must be specified for each bunch and each dblm type'
#     else:
#         baseline = {}
#         for dblmType in ['V','H-V-S']:
#             baseline[dblmType] = np.min(data_dblm[dblmType],axis=0)

#     for dblmType in ['V','H-V-S']:
#         data_dblm[dblmType] -= baseline[dblmType]
#     #---------------------------------



#     eta = (data_dNdt + (data_dblm['V']+data_dblm['H-V-S']))/(data_dNdt)
#     #==================================

#     # UNPACKING RESULTS
#     bin_unix      = (times - df.iloc[0]['Time'])*1e9 + df.index[0]
#     bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]


#     return    pd.DataFrame({'Time':times,'Timestamp':bin_timestamp,
#                             'dNdt'         :list(data_dNdt),
#                             'Losses V'     :list(data_dblm['V']),
#                             'Losses H-V-S' :list(data_dblm['H-V-S']),
#                             'eta'          :list(eta)},index=bin_unix)

