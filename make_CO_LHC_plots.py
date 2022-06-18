import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Make sure this import is after pandas
import Backend.Constants as cst
import Backend.NXCALS as nx
import twissExtractor

# Importing data
database = pd.read_parquet('Data_merged/dataset_2022_05_22.parquet')
database.insert(0,'Time',(database.index - database.index[0]).total_seconds())

MDWindow = [pd.Timestamp('2022-05-22 13:30'),pd.Timestamp('2022-05-22 16:45')]
MDTime = (MDWindow[1]-MDWindow[0]).total_seconds()

#locations = ['L1B1','R1B2','L5B1','R5B2']
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
lineInfo  = [('b','-v'),('r','-v'),('b','-^'),('r','-^')]




# Grouping data into ON-current phases for both B1 and B2
#=========================================================
phase_list= {'B1:1': [pd.Timestamp('2022-05-22 13:30')   ,pd.Timestamp('2022-05-22 14:11')],
             'B1:2': [pd.Timestamp('2022-05-22 15:03')   ,pd.Timestamp('2022-05-22 15:17')],
             'B1:3': [pd.Timestamp('2022-05-22 15:41')   ,pd.Timestamp('2022-05-22 16:06')],
             'B2:1': [pd.Timestamp('2022-05-22 14:12')   ,pd.Timestamp('2022-05-22 14:43')],
             'B2:2': [pd.Timestamp('2022-05-22 15:18')   ,pd.Timestamp('2022-05-22 15:35')],
             'B2:3': [pd.Timestamp('2022-05-22 16:09:30'),pd.Timestamp('2022-05-22 16:30')]
            }

if 'Phase' not in database.columns:
    database.insert(1,'Phase','None')
    for name,_window in phase_list.items():
        cutting = pd.cut(database.index,bins=_window)

        database.loc[list(database.groupby(cutting).get_group(cutting.categories[0]).index),'Phase']  = name
#==========================================================



#============================
# finding beta function @ wire
twiss = twissExtractor.fromOptics('opticsfile.1')

for beam in beams:
    for wire in wires[beam.name]:
        _twissLoc  = [name for name in twiss[beam.name].index if ('bbcw' in name)&(wire.loc[:2].lower() in name)]
        upstream   = [name for name in _twissLoc if '.u.' in name][0]
        downstream = [name for name in _twissLoc if '.d.' in name][0]

        # Averaging beta upstream and downstream
        betx,bety = twiss[beam.name].loc[[upstream,downstream],['betx','bety']].mean()

        # Finding wire length
        #L = np.max(np.abs(twiss[beam.name].loc[[upstream,downstream],['s']].diff()))[0]
        L = 1

        # Attaching to wire class
        wire.betx = betx
        wire.bety = bety
        wire.L    = L



        
emitt  = 2.5e-6/(1+450e9/cst.m_p_eV)

grouped = database.groupby('Phase')
for phase, group in grouped:
    if phase == 'None':
        continue
    
    # SKIPPING PHASE 1
    if ':1' in phase:
        continue
            
    # Selecting proper beam and wires for the group
    beam = [beam for beam in beams if beam.name in phase][0]
    wire1 = wires[beam.name][0]
    wire2 = wires[beam.name][1]
    
    # Finding max index for ref
    I1_start = np.min(np.where(np.abs(group[wire1['I']].dropna())>1))
    I2_start = np.min(np.where(np.abs(group[wire2['I']].dropna())>1))
    
    ref_stop = np.min([group[wire1['I']].dropna().index[I1_start],
                       group[wire2['I']].dropna().index[I2_start]]) #- pd.Timedelta(minutes=1)
    
    ref_start = group.index[0]
    
    
    
    
    nCuts = 100
    
    for num,ts in enumerate(group.index[::len(group.index)//(nCuts-1)]):
            
        
        #fig,axes = MakeCustomGrid()
        fig,axes = Make_LHC_Fig()
        
        for beam in beams:
            plt.sca(axes[beam.name])
            axes[beam.name].set_title(f'Beam {beam.name[-1]}')
            
            ref_BPM  = nx.getBPM_average(ref_start,ref_stop,database,beam)
            extracted_BPM = nx.getBPM_fromTs(ts,database,beam)
        
            for plane in ['H','V']:
                # Extracting BPM values
                _extract = pd.DataFrame({'Value-BPM':extracted_BPM[f'BPMValues_{beam.name}_{plane} [m]']},index=extracted_BPM[f'Mask_{beam.name}_{plane}'])
                _ref     = pd.DataFrame({'Value-Ref':ref_BPM[f'BPMValues_{beam.name}_{plane} [m]']},index=ref_BPM[f'Mask_{beam.name}_{plane}'])

                # Extracting twiss parameter from BPM name

                mu_i,bet_i = {'H':'mux','V':'muy'}[plane], {'H':'betx','V':'bety'}[plane]
                mu,bet = (twiss[beam.name].loc[ref_BPM[f'Mask_{beam.name}_{plane}'],c] for c in [mu_i,bet_i])

                #mu = mu - 
                
                # Putting all the info together to align index (BLM name)
                BPM_df = pd.concat([mu,bet,_ref,_extract],axis=1)
                BPM_df.insert(4,'CO',BPM_df['Value-BPM']-BPM_df['Value-Ref'])
                BPM_df.sort_values(by=mu_i,inplace=True)
        
                # PLOTTING
                meta = f'Mean={np.mean(BPM_df["CO"]/np.sqrt(emitt*BPM_df[bet_i])):.3f} / RMS = {np.std(BPM_df["CO"]/np.sqrt(emitt*BPM_df[bet_i])):.3f}'+r' [$\sigma$]'
                plt.plot(BPM_df[mu_i],BPM_df['CO']/np.sqrt(emitt*BPM_df[bet_i]),label=f'{plane}-position ||'+meta)
                
                _tw_wires = [name for name in twiss[beam.name].index if 'bbcw' in name]
                for _tw_wire in _tw_wires:    
                    plt.axvline(twiss[beam.name].loc[_tw_wire,{'H':'mux','V':'muy'}[plane]],linestyle='--',color={'H':'C0','V':'C1'}[plane])
            
            plt.legend(loc='lower center',fontsize=8)
            plt.xlabel(r'$\mu_x\quad |\quad \mu_y$',fontsize=14)
            plt.ylabel(r'$x/\sqrt{\epsilon\beta_x}\quad |\quad y/\sqrt{\epsilon\beta_y}$',fontsize=14)
            plt.ylim([-0.15,0.15])
        
        # Plotting current
        plt.sca(axes['Current'])
        for wire,color in zip(wires[phase[:2]],['C0','C1']):
            group.nxPlot('index',wire['I'],'-o',ms=2,label = wire.loc + '.' + wire.label['I'])
        plt.axvline(ts,color='k')
        plt.legend(loc='upper left')
        plt.xlabel('UTC Time')
        plt.ylabel(f"{wire.label['I']} [{wire.units['I']}]")
        plt.yticks([-350,-150,150,350])
        
        fig.tight_layout()
        fig.savefig(f'Results_CO/LHC_Phase{phase.replace(":","_")}_TS_{str(num).zfill(3)}.png',format='png',dpi=200)
        plt.close()
    