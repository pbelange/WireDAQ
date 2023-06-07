
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.interpolate as sciInterp

# Deactivating INFO loggings from pynumdiff
#----------------
import logging
logging.basicConfig(level=logging.WARNING)
import pynumdiff
#----------------


import gc

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


def compute_lumi_tot(df,experiments = ['ATLAS','CMS']):

    # Starting with first detector
    obs      = LHC['bb_Luminosity'][experiments[0]]
    lumi_tot = df[obs].dropna().to_frame().rename(columns={obs:'bb_LUMI_TOT'})

    # Adding contribution from other IPs
    times,_tot  = lumi_tot['bb_LUMI_TOT'].to_2D()
    for loc in experiments[1:]:
        times_found,allBunches = df[LHC['bb_Luminosity'][loc]].to_2D()
        _tot   += sciInterp.interp1d(times_found,allBunches,axis=0,bounds_error=False,fill_value=np.nan)(times)

    lumi_tot['bb_LUMI_TOT'] = pd.Series(list(_tot),index=times)
    lumi_tot = lumi_tot.dropna(subset=['bb_LUMI_TOT'])

    # subset = df.dropna(subset='bb_LUMI_TOT')
    return lumi_tot



#----------------------------------
def compute_dBLM_efficiency(df,beams):


    # Extracting data
    #-----------------------------------------
    observable     = 'bb_LUMI_TOT'
    times_L,data_L = df[observable].dropna().to_2D() 

    crop_by   = 10
    data_dblm = {}
    for beam in beams:
        for dblmType in ['V','H-V-S']:
            observable       = beam.dBLM_Amp[dblmType]
            times_dblm,_data = df[observable].dropna().to_2D()
            
            data_dblm[f'{beam.name}:' + dblmType] = _data[crop_by:-crop_by]
    
    times = times_dblm[crop_by:-crop_by]
    #-----------------------------------------


    # Aligning lumi times with interpolation
    #---------------------------------
    data_L    = 1e6*sciInterp.interp1d(times_L,data_L,axis=0,bounds_error=False,fill_value=np.nan)(times)
    #---------------------------------


    # Computing efficiency
    #==================================
    per_beam_results = {}
    for beam in beams:
        dblmTypes = [f'{beam.name}:' + dblmType for dblmType in ['V','H-V-S']]

        # Substracting baseline
        #---------------------------------
        baseline = {}
        for dblmType in dblmTypes:
            # baseline[dblmType] = np.min(data_dblm[dblmType],axis=0)
            baseline[dblmType] = np.mean(np.sort(data_dblm[dblmType][:200,:],axis=0)[50:100,:],axis=0).astype('int32')

        for dblmType in dblmTypes:
            data_dblm[dblmType] -= baseline[dblmType]
            data_dblm[dblmType][data_dblm[dblmType]<= 0] *= -1
        #---------------------------------


        # Computing efficiency
        #=================================
        # R_ellN   = sum([data_dblm[dblmType] for dblmType in dblmTypes])
        R_ellN   = data_dblm[f'{beam.name}:V']+data_dblm[f'{beam.name}:H-V-S']
        eta      = cst.SIG_PP*data_L/(cst.SIG_PP*data_L+R_ellN)
        sig_eff  = cst.SIG_PP/eta
        #==================================

        # Unpacking results
        per_beam_results[f'{beam.name}:eta']     = list(eta)
        per_beam_results[f'{beam.name}:sig_eff'] = list(sig_eff)

    # Removing raw data
    #-----------------------------
    del(data_dblm)
    del(_data)
    del(data_L)
    gc.collect()
    #-----------------------------

    return pd.DataFrame(per_beam_results,index=times)



#=====================
# BCTF
#=====================


def compute_BCT_avg(df,return_raw=False):

    
    idx_max = df[beams[0].Intensity].argmax()
    s_calib = int(df.index[idx_max] + 2*60*1e9)
    e_calib = int(s_calib + 10*60*1e9)

    new_dict = {}
    for beam in beams:

        observable = beam['Intensity']
        ref   = df[observable].dropna()
        times = ref.index 



        observable = beam['bb_Intensity']
        total_A    = df[observable].dropna()
        calib_A    = np.mean((ref/total_A.apply(lambda line: np.sum(line))).loc[s_calib:e_calib])

        new_dict[f'bb_Intensity_A_{beam.name}'] = total_A*calib_A

        observable = beam['bb_Intensity_B']
        total_B    = df[observable].dropna()
        calib_B    = np.mean((ref/total_B.apply(lambda line: np.sum(line))).loc[s_calib:e_calib])

        # B1 BCTFR-B needs rolling of 1 bucket (let's make it general):
        #-------------------------------------
        
        rolling = b_slots[(total_A.loc[s_calib:].iloc[0]>0.5e10)&(b_slots<3562)][-1]-b_slots[(total_B.loc[s_calib:].iloc[0]>0.5e10)&(b_slots<3562)][-1]
        if rolling != 0 :
            total_B = total_B.apply(lambda line: np.roll(line,rolling))
            print(f'MISALIGNED {beam.name}, ROLLING BY {rolling}')
        
        #-------------------------------------

        new_dict[f'bb_Intensity_B_{beam.name}']   = total_B*calib_B

        new_dict[f'bb_Intensity_avg_{beam.name}'] = (new_dict[f'bb_Intensity_A_{beam.name}'] + new_dict[f'bb_Intensity_B_{beam.name}'])/2

    # Removing raw data
    #-----------------------------
    if not return_raw:
        new_dict.pop('bb_Intensity_A_B1')
        new_dict.pop('bb_Intensity_B_B1')
        new_dict.pop('bb_Intensity_A_B2')
        new_dict.pop('bb_Intensity_B_B2')
    #-----------------------------
        del(total_A)
        del(total_B)
        del(ref)
    gc.collect()
    #-----------------------------

    return pd.DataFrame(new_dict)




def compute_BCTF_efficiency(df,beams,smooth_window=None ,smooth_iter=1):

    # Extracting Lumi
    #-----------------------------------------
    observable     = 'bb_LUMI_TOT'
    times_L,data_L = df[observable].dropna().to_2D()
    #-----------------------------------------

    # Fixing time vector
    #---------------------------------
    crop_by   = 10
    _tmp_time = df[f'bb_Intensity_avg_B1'].dropna().index
    dt        = np.min(np.diff(_tmp_time))
    times     = np.arange(_tmp_time[0],_tmp_time[-1],dt)[crop_by:-crop_by]
    #---------------------------------

    # Aligning lumi with interpolation
    #---------------------------------
    data_L  = 1e6*sciInterp.interp1d(times_L,data_L,axis=0,bounds_error=False,fill_value=np.nan)(times)
    #---------------------------------

    per_beam_results = {}
    for beam in beams:

        observable     = f'bb_Intensity_avg_{beam.name}'
        times_N,data_N = df[observable].dropna().to_2D()

        # Computing dNdt
        #---------------------------------
        if smooth_window is not None:
            # Computing dNdt
            dNdt   = np.nan*data_N.copy()
            for bucket in b_slots:
                N_hat,dNdt_hat  = pynumdiff.smooth_finite_difference.meandiff(data_N[:,bucket], dt/1e9, [smooth_window,smooth_iter], options={'iterate': True})
                dNdt[:,bucket]   = dNdt_hat
                data_N[:,bucket] = N_hat 
        else:
            data_N = sciInterp.interp1d(times_N,data_N,axis=0)(times)
            dNdt   = first_order_diff(data_N,dt/1e9)
        #---------------------------------

        # Aligning times with interpolation
        #---------------------------------
        dNdt  =     sciInterp.interp1d(times_N,dNdt,axis=0,bounds_error=False,fill_value=np.nan)(times)
        #---------------------------------

        # Computing efficiency
        #---------------------------------
        eta      = cst.SIG_PP*data_L/(-dNdt)
        sig_eff  = cst.SIG_PP/eta
        #---------------------------------

        # Unpacking results
        per_beam_results[f'{beam.name}:eta']     = list(eta)
        per_beam_results[f'{beam.name}:sig_eff'] = list(sig_eff)


    # Removing raw data
    #-----------------------------
    del(data_L)
    del(data_N)
    del(dNdt)
    gc.collect()
    #-----------------------------

    return pd.DataFrame(per_beam_results,index=times)
