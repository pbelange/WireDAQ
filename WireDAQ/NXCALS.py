
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Function to extrac BPM Values and Mask
#=================================================
def getBPM_average(ts_start,ts_stop,df,beam):
    
    # Ensuring consistent ts format
    if pd.Timestamp(ts_start).tzinfo is None:
        ts_start = pd.Timestamp(ts_start).tz_localize('UTC')
    if pd.Timestamp(ts_stop).tzinfo is None:
        ts_stop = pd.Timestamp(ts_stop).tz_localize('UTC')
        
    subset = getBPM_df(df,beam,ts_start=ts_start,ts_stop=ts_stop)
    
    extracted_BPM =  {}
    for plane in ['H','V']:
        # Extracting BLM names and BLM values
        key_names    = beam[f'BPMNames_{plane}']
        key_selected = beam[f'BPMSelected_{plane}']
        key_values   = beam[f'BPMPosition_{plane}']
    

        # Ensuring that all lengths are compatible for the mean
        vectorLengths = subset[f'BPMValues_{beam.name}_{plane} [m]'].apply(lambda line:len(line))
        vectorLengths.loc[np.abs((vectorLengths-int(np.mean(vectorLengths))))>1] = np.nan
        # returning an error if more than 5% of the data is dropped
        assert(((len(vectorLengths) - len(vectorLengths.dropna()))/len(vectorLengths) <0.05))
        subset = subset.loc[vectorLengths.dropna().index]
    
        extracted_BPM[f'Timestamp']                         = (ts_start,ts_stop)
        extracted_BPM[f'Mask_{beam.name}_{plane}']          = subset[f'Mask_{beam.name}_{plane}'][0]
        extracted_BPM[f'BPMValues_{beam.name}_{plane} [m]'] = subset[f'BPMValues_{beam.name}_{plane} [m]'].mean()
    
    return extracted_BPM
    
def getBPM_fromTs(timestamp,df,beam):
    # Finding NEXT closest timestamp in data
    if pd.Timestamp(timestamp).tzinfo is None:
        timestamp = pd.Timestamp(timestamp).tz_localize('UTC')
        
    indexList = df[beam[f'BPMPosition_H']].dropna().index
    timeDelta = (indexList - timestamp).total_seconds()

    closest = indexList[timeDelta>0][np.argmin(timeDelta[timeDelta>0])]

    extracted_BPM =  {}
    for plane in ['H','V']:
        # Extracting BLM names and BLM values
        key_names    = beam[f'BPMNames_{plane}']
        key_selected = beam[f'BPMSelected_{plane}']
        key_values   = beam[f'BPMPosition_{plane}']

        BPMNames   = df[key_names].dropna().iloc[0]['elements']
        BPMValues  = 1e-6*df[key_values].dropna().loc[closest]['elements']

        # Creating mask from BLMSelected + Beam
        beamFilter = pd.Series(BPMNames).str.contains(f'.{beam.name}')
        selected   = df[key_selected].dropna().loc[closest]['elements']

        mask = np.logical_and(np.array(selected),np.array(beamFilter))

        extracted_BPM[f'Timestamp']                         = closest
        extracted_BPM[f'Mask_{beam.name}_{plane}']          = np.char.lower(BPMNames[mask].astype(str))
        extracted_BPM[f'BPMValues_{beam.name}_{plane} [m]'] = BPMValues[mask]
        

    return extracted_BPM


def getBPM_df(df,beam,ts_start=None,ts_stop=None,loc=''):

    df_BPM = pd.DataFrame({})
    for plane in ['H','V']:
        # Extracting BLM names and BLM values
        key_names    = beam[f'BPMNames_{plane}']
        key_selected = beam[f'BPMSelected_{plane}']
        key_values   = beam[f'BPMPosition_{plane}']
        
        BPMNames   = df[key_names].dropna().iloc[0]['elements'].copy()
        
        if ts_start is not None:
            assert(ts_stop is not None)
            # Ensuring consistent ts format
            if pd.Timestamp(ts_start).tzinfo is None:
                ts_start = pd.Timestamp(ts_start).tz_localize('UTC')
            if pd.Timestamp(ts_stop).tzinfo is None:
                ts_stop = pd.Timestamp(ts_stop).tz_localize('UTC')

            subset = df.loc[ts_start:ts_stop]
        else:
            subset = df
            
        BPMValues  = 1e-6*subset[key_values].dropna().apply(lambda line: np.array(line['elements']))
        
        # Creating mask from BLMSelected + Beam + Loc
        beamFilter = pd.Series(BPMNames).str.contains(rf'(?=.*{beam.name})(?=.*{loc.upper()})',regex=True)
        mask       = subset[key_selected].dropna().apply(lambda line:np.logical_and(line['elements'],np.array(beamFilter)))
        mask.name  = f'Mask_{beam.name}_{plane}'

        # Extracting values for the mask
        extracted = pd.concat([BPMValues,mask],axis=1).apply(lambda line: line[key_values][line[mask.name]],axis=1)
        extracted.name = f'BPMValues_{beam.name}_{plane} [m]'

        # Saving results
        df_BPM = pd.concat([df_BPM,mask.apply(lambda line: np.char.lower(BPMNames[line].astype(str))),extracted],axis=1)
    
    return df_BPM
#=================================================

class NXCALSWire():
    """ Class to keep track of the variable names used in NXCALS for the BBCW 
        
        Run NXCALSWire(loc='L1B1') for an example of the variable names
    
    """
    def __init__(self,name=None,loc=None):
        
        # Names of default wires installed on the collimators
        #---------------------------------------------------
        self._defaults = {'L1B1':'RPMC.UL14.RBBCW.L1B1',
                          'R1B2':'RPMC.UL16.RBBCW.R1B2',
                          'L5B1':'RPMC.USC55.RBBCW.L5B1',
                          'R5B2':'RPMC.UL557.RBBCW.R5B2'} 
        if name is not None:
            self.name = name.upper()
            self.loc  = self.name.split('.')[-1]
        elif loc is not None:
            self.loc  = loc.upper()
            self.name = self._defaults[self.loc]
        else:
            raise ValueError('name or loc has to be specified. Defaults values can be taken from: '+str(self._defaults))

        
        # Variable names for measured quantities:
        #---------------------------------------------------
        
        # Current
        self.I = f'{self.name}:I_MEAS'                                
        # Voltage
        self.V = f'{self.name}:V_MEAS'    
        # TCT name
        self.TCT = {'L1B1':'TCTPV.4L1.B1',
                    'R1B2':'TCTPV.4R1.B2',
                    'L5B1':'TCTPH.4L5.B1',
                    'R5B2':'TCTPH.4R5.B2'}[self.loc]
        # Pressure
        self.Gauge = f'VGPB.935.4{self.loc[:2]}.B.PR'
        # Temperature
        _temp  = self.TCT.replace('.','_')
        self.T = {'Right Upstream'  :f'{_temp}_TTRU.POSST',
                  'Left Upstream'   :f'{_temp}_TTLU.POSST',
                  'Right Downstream':f'{_temp}_TTRD.POSST',
                  'Left Downstream' :f'{_temp}_TTLD.POSST'}
        # Position
        self.Position = {'Right Upstream'  :f'{self.TCT}:MEAS_LVDT_RU',
                         'Left Upstream'   :f'{self.TCT}:MEAS_LVDT_LU',
                         'Right Downstream':f'{self.TCT}:MEAS_LVDT_RD',
                         'Left Downstream' :f'{self.TCT}:MEAS_LVDT_LD'}
        
        # WIC
        self.WIC_A = f'RBBCW.{self.loc}:CMD_ABORT_A_WIC'
        self.WIC_B = f'RBBCW.{self.loc}:CMD_ABORT_B_WIC'
        
        # Keeping track of the units
        self.units = {'I'       :'A',
                      'V'       :'V',
                      'T'       :r'$^\circ$C',
                      'Gauge'   :'mbar',
                      'Position':'mm',
                      'WIC_A'   :'Bool',
                      'WIC_B'   :'Bool'}
        
        self.label = {'I'       :'Current',
                      'V'       :'Voltage',
                      'T'       :'Temperature',
                      'Gauge'   :'Pressure',
                      'Position':'Position',
                      'WIC_A'   :'WIC-A Status',
                      'WIC_B'   :'WIC-B Status'}
        
    def __getitem__(self, item):
        return getattr(self, item)
    
    def _getVarList(self,subset = None):
        varlist = []
        if subset is None:
            lookfor = dir(self)
        else:
            lookfor = subset
        for attr in lookfor:
            if (attr[0]!='_')&(attr not in 'label,loc,name,units'):
                if type(self[attr]) is dict:
                    varlist += list(self[attr].values())
                else:
                    varlist += [self[attr]]
        return varlist
    
    def __repr__(self,):
        rep  = 40*'*' + '\n'
        rep += f'Wire: {self.name}\n'
        rep += f'\t Location: {self.loc}\n'
        rep += f'\t Current : {self.I}\n'
        rep += f'\t Voltage : {self.V}\n'
        rep += f'\t TCT     : {self.TCT}\n'
        rep += f'\t Temp.   : {str(self.T)}\n'
        rep += f'\t Position: {str(self.Position)}\n'
        rep += f'\t WIC_A   : {self.WIC_A}\n'
        rep += f'\t WIC_B   : {self.WIC_B}\n'
        rep += 40*'*' + '\n'
        return rep

        
class NXCALSLHC():

    def __init__(self,name=None):
        
        self.name = 'LHC'
        
        # LUMI
        #---------------------------------------------------
        self.Luminosity    = { 'ATLAS' : 'ATLAS:LUMI_TOT_INST',
                               'ALICE' : 'ALICE:LUMI_TOT_INST',
                               'CMS'   : 'CMS:LUMI_TOT_INST',
                               'LHCB'  : 'LHCB:LUMI_TOT_INST'}
        self.bb_Luminosity = { 'ATLAS' : 'ATLAS:BUNCH_LUMI_INST',
                               'ALICE' : 'ALICE:BUNCH_LUMI_INST',
                               'CMS'   : 'CMS:BUNCH_LUMI_INST',
                               'LHCB'  : 'LHCB:BUNCH_LUMI_INST'}


        # Crossing angles
        self.Xing      = {  'IP1' : 'LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:target',
                            'IP5' : 'LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:target'}


        self.betastar  = {  'IP1' :'HX:BETASTAR_IP1',
                            'IP2' :'HX:BETASTAR_IP2',
                            'IP5' :'HX:BETASTAR_IP5',
                            'IP8' :'HX:BETASTAR_IP8'}

        self.Fill       = 'HX:FILLN'
        self.Inj_Scheme = 'LHC.STATS:LHC:INJECTION_SCHEME'
        self.Bmode      = 'HX:BMODE'
        
        # BPM
        #---------------------------------------------------
        self.BPMPosition_H    = 'BFC.LHC:OrbitAcq:positionsH'
        self.BPMPosition_V    = 'BFC.LHC:OrbitAcq:positionsV'
        
        self.BPMSelected_H = 'BFC.LHC:OrbitAcq:bpmSelectedH'
        self.BPMSelected_V = 'BFC.LHC:OrbitAcq:bpmSelectedV'
        
        self.BPMNames_H    = 'BFC.LHC:Mappings:fBPMNames_h'
        self.BPMNames_V    = 'BFC.LHC:Mappings:fBPMNames_v'
        
        
        # Keeping track of the units
        self.units = {'Luminosity' : 'Hz/ub',
                      'Xing'       : 'urad'}
        
        self.label = {'Luminosity' : 'Luminosity',
                      'Xing'       : 'Xing'}
        
    def __getitem__(self, item):
        return getattr(self, item)

    def _getVarList(self,subset = None):
        varlist = []
        if subset is None:
            lookfor = dir(self)
        else:
            lookfor = subset
        for attr in lookfor:
            if (attr[0]!='_')&(attr not in 'label,loc,name,units'):
                if type(self[attr]) is dict:
                    varlist += list(self[attr].values())
                else:
                    varlist += [self[attr]]
        return varlist
           

            
class NXCALSBeam():
    """ Class to keep track of the variable names used in NXCALS for the Beam 
        
        
    
    """
    def __init__(self,name=None):
        
        self.name = name.upper()


        # Bunches
        self.Nb = f'LHC.BQM.{self.name}:NO_BUNCHES'

        # Filled Buckets
        self.Filled_b = f'LHC.BQM.{self.name}:FILLED_BUCKETS'

        # Intensity
        self.Intensity    = f'LHC.BCTDC.A6R4.{self.name}:BEAM_INTENSITY'
        self.bb_Intensity = f'LHC.BCTFR.A6R4.{self.name}:BUNCH_INTENSITY'

        self.Intensity_B    = f'LHC.BCTDC.B6R4.{self.name}:BEAM_INTENSITY'
        self.bb_Intensity_B = f'LHC.BCTFR.B6R4.{self.name}:BUNCH_INTENSITY'


        # Emittance
        _loc_fmt = {'B1':'5R4.B1','B2':'5L4.B2'}[self.name]
        self.bb_Emittance_H = f'LHC.BSRT.{_loc_fmt}:BUNCH_EMITTANCE_H'
        self.bb_Emittance_V = f'LHC.BSRT.{_loc_fmt}:BUNCH_EMITTANCE_V'

        # Tune
        self.Qx = f'LHC.BQBBQ.CONTINUOUS.{self.name}:TUNE_H'
        self.Qy = f'LHC.BQBBQ.CONTINUOUS.{self.name}:TUNE_V'
        
        self.Qx_HS = f'LHC.BQBBQ.CONTINUOUS_HS.{self.name}:TUNE_H'
        self.Qy_HS = f'LHC.BQBBQ.CONTINUOUS_HS.{self.name}:TUNE_V'
       

        # dBLM 
        _loc_fmt = {'B1':'.3','B2':'.5'}[self.name]
        _dBLM    = {    'V'     : f'HC.TZ76.BLMDIAMOND2{_loc_fmt}',
                        'H-V-S' : f'HC.TZ76.BLMDIAMOND3{_loc_fmt}'}
        self.dBLM     = {   'V'     : f'{_dBLM["V"]}:AcquisitionIntegral:intSumBuf0',
                            'H-V-S' : f'{_dBLM["H-V-S"]}:AcquisitionIntegral:intSumBuf0'}
        self.dBLM_Amp = {   'V'     : f'{_dBLM["V"]}:AcquisitionIntegral:intSumBuf1',
                            'H-V-S' : f'{_dBLM["H-V-S"]}:AcquisitionIntegral:intSumBuf1'}

    
        # Keeping track of the units
        self.units = {'Qx'      :'',
                      'Qy'      :'',
                      'Qx_HS'   :'',
                      'Qy_HS'   :'',
                      'Luminosity': 'Hz/ub',
                      'Intensity':'protons'}
        
        self.label = {'Qx'      :r'$Q_x$',
                      'Qy'      :r'$Q_y$',
                      'Qx_HS'   :r'$Q_x$',
                      'Qy_HS'   :r'$Q_y$',
                      'Luminosity': 'Luminosity',
                      'Intensity':'Intensity'}
        
    def __getitem__(self, item):
        return getattr(self, item)

    def _getVarList(self,subset = None):
        varlist = []
        if subset is None:
            lookfor = dir(self)
        else:
            lookfor = subset
        for attr in lookfor:
            if (attr[0]!='_')&(attr not in 'label,loc,name,units'):
                if type(self[attr]) is dict:
                    varlist += list(self[attr].values())
                else:
                    varlist += [self[attr]]
        return varlist
    
    def __repr__(self,):
        rep  = 40*'*' + '\n'
        rep += f'Beam: {self.name}\n'
        rep += f'\t Intensity: {self.Intensity}\n'
        rep += f'\t Tune x   : {self.Qx}\n'
        rep += f'\t Tune y   : {self.Qy}\n'
        rep += 40*'*' + '\n'
        return rep
    
    
    
    
