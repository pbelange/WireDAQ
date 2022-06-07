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
    
    def _getVarList(self):
        varlist = []
        for attr in dir(self):
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

        
                    

            
class NXCALSBeam():
    """ Class to keep track of the variable names used in NXCALS for the Beam 
        
        
    
    """
    def __init__(self,name=None):
        
        self.name = name.upper()
        
        # Variable names for measured quantities:
        #---------------------------------------------------
        
        # Intensity
        self.Intensity = f'LHC.BCTDC.A6R4.{self.name}:BEAM_INTENSITY'
        
        # BPM
        self.Position_H = 'BFC.LHC:OrbitAcq:positionsH'
        self.Position_V = 'BFC.LHC:OrbitAcq:positionsV'
        
        self.BPM_H      = 'BFC.LHC:OrbitAcq:bpmSelectedH'
        self.BPM_V      = 'BFC.LHC:OrbitAcq:bpmSelectedV'
        
        
        # Tune
        self.Qx = f'LHC.BQBBQ.CONTINUOUS.{self.name}:TUNE_H'
        self.Qy = f'LHC.BQBBQ.CONTINUOUS.{self.name}:TUNE_V'
        
        self.Qx_HS = f'LHC.BQBBQ.CONTINUOUS_HS.{self.name}:TUNE_H'
        self.Qy_HS = f'LHC.BQBBQ.CONTINUOUS_HS.{self.name}:TUNE_V'
       
    
        # Keeping track of the units
        self.units = {'Qx'      :'',
                      'Qy'      :'',
                      'Qx_HS'   :'',
                      'Qy_HS'   :''}
        
        self.label = {'Qx'      :r'$Q_x$',
                      'Qy'      :r'$Q_y$',
                      'Qx_HS'   :r'$Q_x$',
                      'Qy_HS'   :r'$Q_y$'}
        
    def __getitem__(self, item):
        return getattr(self, item)

    def _getVarList(self):
        varlist = []
        for attr in dir(self):
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
    
    
    
    
