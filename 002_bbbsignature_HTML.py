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


# Setting default values
#------------------------------------------------
_default_fig_width  = 2000
_default_fig_height = 400
_default_fig_pad    = 100

_default_device = 'DBLM'

_default_path     = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2023/'
_default_out      = f'/eos/user/p/phbelang/www/Monitoring_bbbsignature/{_default_device}'


def Efficiency_to_HTML(FILL, HTML_name=None,data_path=_default_path,device = _default_device):


    assert HTML_name is not None, 'Please provide a name for the HTML file'



    print('\n' + 40*'=')
    print('IMPORTING DATA')
    database,bb_df_b1,bb_df_b2 = run_analysis(FILL,data_path=data_path,device = device)
    

    print(40*'-')
    print('CREATING FIGURES')
    # Creating figures
    #=====================================

    BOKEH_FIGS = {}


    # Making data source for most plots
    #-------------------------------------
    dt_I          = 5*60
    default_ts    = database['Timestamp'].iloc[len(database)//2]
    left_ts       = default_ts - pd.Timedelta(minutes=5)
    right_ts      = default_ts + pd.Timedelta(minutes=5)
    
    source,data_I,data_E_H,data_E_V,data_bb = make_efficiency_source(database,bb_df_list=[bb_df_b1,bb_df_b2],dt_I = dt_I,left=left_ts,right=right_ts)
    #-------------------------------------

    # Overview plot
    #-------------------------------------
    BOKEH_FIGS['Overview'] = make_overview_figure(FILL,database)

    # Add slider tool
    #-------------------------------------
    slider_colors = {'left':'coral','right':'mediumseagreen','avg':'slateblue'}
    time_slider = Windower(BOKEH_FIGS['Overview'],
                        left     = left_ts,
                        right    = right_ts,
                        colors   = slider_colors,
                        lw       = pd.Timedelta(minutes=0.5),
                        box_kwargs = dict(fill_color='black',fill_alpha=0.2,line_alpha=0,line_width=20),
                        bar_kwargs = dict(fill_alpha=0.5,line_width=2,line_alpha=0.5))

    time_slider.add_start_callback(args = dict( sourceb1= source['B1'],
                                                sourceb2= source['B2'],
                                                ts_list = list(data_bb['B1'].index),
                                                ts_I_list = list(data_I['B1'].index),
                                                bb_b1   = data_bb['B1'],
                                                bb_b2   = data_bb['B2'],
                                                bb_I_b1 = data_I['B1'],
                                                bb_I_b2 = data_I['B2'],
                                                bb_EH_b1 = data_E_H['B1'],
                                                bb_EH_b2 = data_E_H['B2'],
                                                bb_EV_b1 = data_E_V['B1'],
                                                bb_EV_b2 = data_E_V['B2']),
                                                code    =  """
        //====================================================
        //--------------------------------
        function addvector(a,b){
            return a.map((e,i) => e + b[i]);
        }

        function divvector(a,b){
            return a.map((e,i) => e/b);
        }

        function average2D(array) {
            return divvector(array.reduce((x,y) => addvector(x,y)),array.length);
        }
        //--------------------------------

        // Finding index of closest timestamp
        const _s     = cb_obj.start-metadata.data['lw'][0]/2;
        const _s_idx = ts_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_s)));
        const _e     = cb_obj.end+metadata.data['lw'][0]/2;
        const _e_idx = ts_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_e)));
        const _m     = cb_obj.start+(cb_obj.end-cb_obj.start)/2;
        const _m_idx = ts_I_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_m)));

        sourceb1.data.data_l = bb_b1[_s_idx];
        sourceb2.data.data_l = bb_b2[_s_idx];

        sourceb1.data.Intensity = bb_I_b1[_m_idx];
        sourceb2.data.Intensity = bb_I_b2[_m_idx];
        sourceb1.data.Emitt_H = bb_EH_b1[_m_idx];
        sourceb2.data.Emitt_H = bb_EH_b2[_m_idx];
        sourceb1.data.Emitt_V = bb_EV_b1[_m_idx];
        sourceb2.data.Emitt_V = bb_EV_b2[_m_idx];
        
        sourceb1.data.data_avg = average2D(bb_b1.slice(_s_idx,_e_idx));
        sourceb2.data.data_avg = average2D(bb_b2.slice(_s_idx,_e_idx));

        sourceb1.change.emit();
        sourceb2.change.emit();
        //====================================================""")

    time_slider.add_end_callback(args = dict(   sourceb1= source['B1'],
                                                sourceb2= source['B2'],
                                                ts_list = list(data_bb['B1'].index),
                                                ts_I_list = list(data_I['B1'].index),
                                                bb_b1   = data_bb['B1'],
                                                bb_b2   = data_bb['B2'],
                                                bb_I_b1 = data_I['B1'],
                                                bb_I_b2 = data_I['B2'],
                                                bb_EH_b1 = data_E_H['B1'],
                                                bb_EH_b2 = data_E_H['B2'],
                                                bb_EV_b1 = data_E_V['B1'],
                                                bb_EV_b2 = data_E_V['B2']),
                                                code    =  """
        //====================================================
        //--------------------------------
        function addvector(a,b){
            return a.map((e,i) => e + b[i]);
        }

        function divvector(a,b){
            return a.map((e,i) => e/b);
        }

        function average2D(array) {
            return divvector(array.reduce((x,y) => addvector(x,y)),array.length);
        }
        //--------------------------------

        // Finding index of closest timestamp
        const _s     = cb_obj.start-metadata.data['lw'][0]/2;
        const _s_idx = ts_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_s)));
        const _e     = cb_obj.end+metadata.data['lw'][0]/2;
        const _e_idx = ts_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_e)));
        const _m     = cb_obj.start+(cb_obj.end-cb_obj.start)/2;
        const _m_idx = ts_I_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_m)));

        sourceb1.data.data_r = bb_b1[_e_idx];
        sourceb2.data.data_r = bb_b2[_e_idx];

        sourceb1.data.Intensity = bb_I_b1[_m_idx];
        sourceb2.data.Intensity = bb_I_b2[_m_idx];
        sourceb1.data.Emitt_H = bb_EH_b1[_m_idx];
        sourceb2.data.Emitt_H = bb_EH_b2[_m_idx];
        sourceb1.data.Emitt_V = bb_EV_b1[_m_idx];
        sourceb2.data.Emitt_V = bb_EV_b2[_m_idx];
        
        sourceb1.data.data_avg = average2D(bb_b1.slice(_s_idx,_e_idx));
        sourceb2.data.data_avg = average2D(bb_b2.slice(_s_idx,_e_idx));

        sourceb1.change.emit();
        sourceb2.change.emit();
        //====================================================""")


    # Efficiency plot
    #-------------------------------------
    for beam,bb_df,color in zip(beams,[bb_df_b1,bb_df_b2],['royalblue','firebrick']):
        BOKEH_FIGS[f'BBBSignature {beam.name}'],BOKEH_FIGS[f'widget {beam.name}'] = make_bbbsignature_figure(source,beam,bb_df,colors = slider_colors)

        # BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.background_fill_color = color
        # BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.background_fill_alpha = 0.5
        # BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.label_text_color      = "white"
        # BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.title_text_color      = "white"
        # BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.title_text_font_style = "bold"


        BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.title_text_font_style = "bold"
        BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.border_line_color     = color
        BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.border_line_width     = 4
        BOKEH_FIGS[f'BBBSignature {beam.name}'].legend.border_line_alpha     = 0.9

        BOKEH_FIGS[f'BBBSignature {beam.name}'].outline_line_width = 4
        BOKEH_FIGS[f'BBBSignature {beam.name}'].outline_line_alpha = 0.4
        BOKEH_FIGS[f'BBBSignature {beam.name}'].outline_line_color = color

    # Intensity
    #-------------------------------------
    for beam,bb_df,color in zip(beams,[bb_df_b1,bb_df_b2],['royalblue','firebrick']):
        BOKEH_FIGS[f'Intensity {beam.name}'] = make_intensity_figure(source,data_I,beam,color)#make_intensity_figure(FILL,database,source,data_I,beam,bb_df,color)

    # Emittance
    #-------------------------------------
    for beam,bb_df,color in zip(beams,[bb_df_b1,bb_df_b2],['royalblue','firebrick']):
         for data_E,plane in zip([data_E_H,data_E_V],['H','V']):
            BOKEH_FIGS[f'Emitt_{plane} {beam.name}'] = make_emittance_figure(source,data_E,plane,beam,color)
    #=====================================
    




    print(40*'-')
    print('Exporting to HTML')
    # LAYOUT
    #=====================================

    # Adjusting canvas size:
    for name,_fig in BOKEH_FIGS.items():
        if 'widget' in name.lower():
            continue
        _fig.min_border_left  = _default_fig_pad
        _fig.min_border_right = _default_fig_pad


    # Linking x-axis for bunch-by-bunch plots
    #---------------------------------
    # In X
    BOKEH_FIGS[f'BBBSignature B1'].x_range = BOKEH_FIGS[f'Intensity B1'].x_range
    BOKEH_FIGS[f'BBBSignature B2'].x_range = BOKEH_FIGS[f'Intensity B1'].x_range

    # In Y
    BOKEH_FIGS[f'BBBSignature B2'].y_range = BOKEH_FIGS[f'BBBSignature B1'].y_range

    # Making tabs
    BOKEH_FIGS[f'BBBSignature B1B2'] = bkmod.Tabs(tabs=[  bkmod.TabPanel(child=bklay.row(BOKEH_FIGS[f'BBBSignature B1'],BOKEH_FIGS[f'widget B1']), title="Beam 1"), 
                                                        bkmod.TabPanel(child=bklay.row(BOKEH_FIGS[f'BBBSignature B2'],BOKEH_FIGS[f'widget B2']), title="Beam 2")])


    # Linking both intensities and making grid
    #---------------------------------
    BOKEH_FIGS[f'Intensity B2'].x_range = BOKEH_FIGS[f'Intensity B1'].x_range
    BOKEH_FIGS[f'Intensity B2'].y_range = BOKEH_FIGS[f'Intensity B1'].y_range

    for beam in beams:
        for plane in ['H','V']:
            BOKEH_FIGS[f'Emitt_{plane} {beam.name}'].x_range = BOKEH_FIGS[f'Intensity B1'].x_range
            BOKEH_FIGS[f'Emitt_{plane} {beam.name}'].y_range = BOKEH_FIGS[f'Emitt_H B1'].y_range


    # Making tabs
    BOKEH_FIGS[f'Int. and Emitt'] = bkmod.Tabs(tabs=[  bkmod.TabPanel(child=bklay.column(BOKEH_FIGS['Intensity B1'],BOKEH_FIGS['Intensity B2']), title="Intensity"), 
                                                       bkmod.TabPanel(child=bklay.column(BOKEH_FIGS['Emitt_H B1'],BOKEH_FIGS['Emitt_H B2'])    , title="Emittance H"),
                                                       bkmod.TabPanel(child=bklay.column(BOKEH_FIGS['Emitt_V B1'],BOKEH_FIGS['Emitt_V B2'])    , title="Emittance V")])


    # BOKEH_FIGS['Intensity B1B2'] =  bklay.gridplot([[BOKEH_FIGS['Intensity B1']], [BOKEH_FIGS['Intensity B2']]])#,toolbar_location='right')
    
    
    # Making final layout
    #---------------------------------
    HTML_LAYOUT = bklay.column(BOKEH_FIGS['Overview'],
                               bklay.gridplot([[BOKEH_FIGS[f'BBBSignature B1B2']],
                                               [BOKEH_FIGS[f'Int. and Emitt']]],toolbar_location='right'))
    #=====================================

    # Removing raw data
    #-----------------------------
    del(database)
    gc.collect()
    #-----------------------------


    # Exporting to HTML
    #=====================================
    export_HTML(HTML_LAYOUT,HTML_name,f'Fill {FILL}')
    print(40*'=')

    
    # return database,source,data_I



def run_analysis(FILL,data_path=_default_path,device = _default_device):

    # Fixing data path
    # raw_data    = data_path + '/rawdata/'
    # device_data = data_path + f'/efficiency_data/{device}/'

    raw_data    = data_path + '/rawdata/'
    device_data = data_path + f'../2023/efficiency_data/{device}/'



    # Finding filling pattern
    #-------------------------------------------------
    bb_df_b1,bb_df_b2 = parser.fill_filling_pattern(fill=FILL,data_path= raw_data,n_LR = 21)
    #-------------------------------------------------



    # Declaring master bin times
    #-------------------------------------------------
    dt = 60
    unix_s,unix_e = parser.fill_unix_times(FILL,data_path=raw_data)
    unix_bins     = np.arange(unix_s,unix_e,dt/1e-9)
    #-------------------------------------------------

    # Import efficiency
    #-------------------------------------------------
    # try:
    df_eff = parser.from_parquet2bin(file= device_data + f'/FILL{FILL}.parquet',bins=unix_bins)
    # except:
    #     print('!!! Failed to import dBLM,trying to drop first and last 10% of data !!!')
    #     unix_bins = unix_bins[len(unix_bins)//10:-len(unix_bins)//10]
    #     df_eff = parser.from_parquet2bin(file= device_data + f'/FILL{FILL}.parquet',bins=unix_bins)
    #-------------------------------------------------


    # Import lumi and compute total lumi
    #-------------------------------------------------
    variables  = [LHC.bb_Luminosity['ATLAS'],LHC.bb_Luminosity['CMS']]
    df_lumi     = parser.from_parquet(fill=FILL,variables = variables,beamMode = None,data_path= raw_data)
    df_lumi_tot = eff.compute_lumi_tot(df_lumi,experiments = ['ATLAS','CMS'])
    #-------------------------------------------------

    # Cropping Efficiency
    #------------------------------------------------- 
    # droping near-zero values
    # n_colliding   = len(bb_df_b1['HO partner in ATLAS'].dropna())
    # avg_per_bunch = df_lumi_tot['bb_LUMI_TOT'].apply(lambda line:np.sum(line)/n_colliding)
    # noise_level   = 0.00115

    # ROI_idx = avg_per_bunch[avg_per_bunch>5*noise_level].index[[0,-1]]
    # df_eff  = df_eff.loc[ROI_idx[0]:ROI_idx[1]]
    #----------------------------------



    # Adding extra data for analysis
    #-------------------------------------------------
    LHC_vars   = LHC._getVarList(subset=[   'Xing',
                                            'Fill',
                                            'Bmode',
                                            'betastar'])


    beams_vars = sum([beam._getVarList(subset=[ 'Intensity',
                                                'Nb']) for beam in beams],[])
    wires_vars = sum([wire._getVarList(subset=[ 'I']) for wire in wires['B2']],[])
    variables = sorted(LHC_vars+beams_vars+wires_vars)

    df_extra = parser.from_parquet(fill=FILL,variables = variables,beamMode = None,data_path= raw_data)

    # Extra bunch-by-bunch data
    #-------------------------------------------------
    variables = sum([beam._getVarList(subset=['bb_Intensity','bb_Emittance_H','bb_Emittance_V']) for beam in beams],[])
    unix_bins_I  = np.arange(unix_s,unix_e,5*60/1e-9)
    df_intensity = parser.from_parquet2bin(fill=FILL,variables = variables,bins=unix_bins_I,beamMode = None,data_path= raw_data)



    database = pd.concat([df_extra,df_intensity,df_lumi,df_eff])
    database = parser.make_timestamps(database)

    # Removing raw data
    #-----------------------------
    del(df_lumi_tot)
    del(df_extra)
    del(df_intensity)
    del(df_lumi)
    del(df_eff)
    gc.collect()
    #-----------------------------

    return database,bb_df_b1,bb_df_b2





# New axis function
#=====================================
def new_axis(fig,axis_name,side='none'):
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1)
    _ax = bkmod.LinearAxis(y_range_name=axis_name)
    if side == 'none':
        pass
    else:
        fig.add_layout(_ax,side)

    return _ax,axis_name
#=====================================

def make_overview_figure(FILL,database):
    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = _default_fig_height, 
                    width           = _default_fig_width,
                    title           = "Overview" + f' FILL {FILL:d}  ({database["Timestamp"].iloc[0].strftime("%Y-%m-%d")})', 
                    x_axis_type     = "datetime",
                    tools           = "pan,box_zoom,reset,save,hover",
                    active_drag     = "box_zoom",
                    x_range         = bkmod.Range1d(database["Timestamp"].iloc[0],database["Timestamp"].iloc[-1],bounds='auto'))
    fig.xaxis.formatter= bkmod.DatetimeTickFormatter(hourmin = '%H:%M',hours='%H:%M',days='%H:%M',months='%H:%M',years='%H:%M')

    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]

    # fig.tags[0]['PanTool'].update(dimensions = 'width')
    fig.tags[0]['HoverTool'].update(tooltips=[('Variable', '$name'),('Time (H:M)','$x{%H:%M}'),('Value','$y')],formatters={ "$x": "datetime"},muted_policy='ignore')
    #=====================================




    # Plotting Intensity
    #--------------------
    for beam,color in zip(beams,['blue','red']):
        data = database.set_index('Timestamp')[beam.Intensity].dropna()
        
        fig.line(data.index,data,color=color,alpha=0.8,legend_label=beam.name,name=f'Intensity {beam.name} [p+]')

    fig.yaxis.axis_label = "Intensity [p+]"
    fig.xaxis.axis_label = f"Local Time, {database['Timestamp'].iloc[0].strftime('%Y-%m-%d')}"
    #--------------------

    # Plotting Luminosity
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='Luminosity')#,side='left')
    max_y = 0
    for loc,color in zip(['ATLAS','CMS'],['orange','green']):
        data = database.set_index('Timestamp')[LHC['bb_Luminosity'][loc]].dropna()
        data = data.apply(lambda line: np.sum(line))
        
        fig.line(data.index,data,color=color,alpha=0.8,legend_label=loc,name=f'Luminosity {loc} [Hz/ub]',y_range_name=axis_name)

        max_y  = np.max((max_y,np.max(data)))

    fig.extra_y_ranges[axis_name] = bkmod.Range1d(-0.05*max_y,1.05*max_y)
    # ax.axis_label = r"Luminosity [Hz/ub]"
    #--------------------


    # Plotting xing
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='xing')#,side='right')


    data = database.set_index('Timestamp')[LHC.Xing['IP5']].dropna()
    fig.line(data.index,data,color='indianred',alpha=0.8,legend_label="theta/2",name="theta/2 [urad]",y_range_name=axis_name)

        
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(110,180)
    # ax.axis_label = r"Half-crossing angle [urad]"
    #--------------------


    # Plotting beta star
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='beta')#,side='right')


    data = database.set_index('Timestamp')[LHC.betastar['IP5']].dropna()
    data[(data>90)|(data<0)]=np.nan
    fig.line(data.index,data,color='purple',alpha=0.8,legend_label="beta*",name="beta* [cm]",y_range_name=axis_name)

        
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,90)
    # ax.axis_label = r"Beta star [cm]"
    #--------------------




    # Plotting wire current
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='wire')#,side='right')

    for wire in wires['B2'] :
        data = database.set_index('Timestamp')[wire.I].dropna()

        fig.line(data.index,data,color='teal',alpha=0.8,legend_label="BBCW",name="BBCW [A]",y_range_name=axis_name)

    max_y = 2000
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(-0.05*max_y,1.05*max_y)
    # ax.axis_label = r"BBCW Current [A]"
    #--------------------

    # Legend Options
    #=====================================
    fig.legend.location     = "top_left"
    fig.legend.click_policy = "hide"
    #=====================================

    return fig


def make_efficiency_source(database,bb_df_list=None,dt_I = 300,left=None,right=None):
    # Creating shared source for plots
    #=====================================================================
    source  = {}
    data_I  = {}
    data_E_H= {}
    data_E_V= {}
    data_bb = {}
    for beam,bb_df in zip(beams,bb_df_list):
        
        
        # Extracting efficiency
        observable   = f'{beam.name}:eta'
        _times,_data = database.set_index('Timestamp')[observable].dropna().to_2D()
        _times       = _times.to_list()

        # Computing bunch-by-bunch data
        observable        = f'{beam.name}:eta'
        _data_bb          = database.set_index('Timestamp')[observable].dropna()
        _data_bb          = 1/_data_bb - 1
        data_bb[beam.name] = _data_bb.apply(lambda line: line[bb_df.index])

        # Computing average intensity and emittance
        _data_I           = database.set_index('Timestamp')[beam.bb_Intensity].dropna()
        data_I[beam.name] = _data_I.groupby(pd.Grouper(freq=f'{dt_I}s')).mean().apply(lambda line: line[bb_df.index]/1e11)

        _data_           = database.set_index('Timestamp')[beam.bb_Emittance_H].dropna()
        data_E_H[beam.name] = _data_.groupby(pd.Grouper(freq=f'{dt_I}s')).mean().apply(lambda line: line[bb_df.index])

        _data_           = database.set_index('Timestamp')[beam.bb_Emittance_V].dropna()
        data_E_V[beam.name] = _data_.groupby(pd.Grouper(freq=f'{dt_I}s')).mean().apply(lambda line: line[bb_df.index])


        # Compiling data to source
        _source_df   = pd.DataFrame({   'Bunch'     :bb_df.index,
                                        'Intensity' :data_I[beam.name][left + (right-left)/2:][0],
                                        'Emitt_H'   :data_E_H[beam.name][left + (right-left)/2:][0],
                                        'Emitt_V'   :data_E_V[beam.name][left + (right-left)/2:][0],
                                        'data_l'    :data_bb[beam.name][left:][0],
                                        'data_avg'  :data_bb[beam.name][left:right].mean(),
                                        'data_r'    :data_bb[beam.name][right:][0],
                                        'Timestamp' :[_times       for idx in bb_df.index],
                                        'eta'       :[_data[:,idx] for idx in bb_df.index]})
        
        source[beam.name] = bkmod.ColumnDataSource(_source_df)
    #=====================================================================

    return source,data_I,data_E_H,data_E_V,data_bb



class Windower():
    def __init__(self,_fig,left,right,colors = {'left':'coral','right':'mediumseagreen'},lw=pd.Timedelta(minutes=2),box_kwargs = {},bar_kwargs = {}):
        
        # Icon file
        self.icon     = Path('WireDAQ/Icons/slider_icon.png')

        # Creating new axis from 0-1 on the figure
        ax,axis_name = new_axis(_fig,axis_name='_slider')
        _fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1)
        bar_kwargs.update({'y_range_name':axis_name})

        # Metadata
        w_factor = 3
        self.metadata  = bkmod.ColumnDataSource({'left': [left],'right':[right],'lw':[lw] ,'yt':[2],'yb':[-1],'ym':[0.5]})

        # Range object
        self.range = bkmod.Range1d(left+lw/2,right-lw/2)


        # Add tool to figure
        self.RangeTool = bkmod.RangeTool(   x_range      = self.range,
                                            icon         = self.icon)
        
        box_kwargs.update({'level':'underlay'})
        self.RangeTool.overlay.update(**box_kwargs)
        #       visible      = False,
        #                                     fill_color   = None,
        #                                     fill_alpha   = 0,
        #                                     line_color   = None)
        _fig.add_tools(self.RangeTool)


        # Add visible slider itself
        self.renderer_l = _fig.rect(x="left"    , y="ym", width="lw", height="yt", source=self.metadata,line_color = colors['left'] ,fill_color=colors['left'] ,**bar_kwargs)
        self.renderer_r = _fig.rect(x="right"   , y="ym", width="lw", height="yt", source=self.metadata,line_color = colors['right'],fill_color=colors['right'],**bar_kwargs)


        # Muting for overlay tool
        self.renderer_l.muted = True
        self.renderer_r.muted = True
        self.renderer_l.muted_glyph = self.renderer_l.glyph.clone()
        self.renderer_r.muted_glyph = self.renderer_r.glyph.clone()


        # Default callback to link the range with rendered object
        #-------------------------
        self.callback_start_0 = bkmod.callbacks.CustomJS(args=dict(metadata = self.metadata), 
                                                                            code="""
                //=========================================================
                metadata.data['left'] = [cb_obj.start-metadata.data['lw'][0]/2];
                metadata.change.emit()
                //=========================================================""")

        self.callback_end_0 = bkmod.callbacks.CustomJS(args=dict(metadata = self.metadata), 
                                                                            code="""
                //=========================================================
                metadata.data['right'] = [cb_obj.end+metadata.data['lw'][0]/2];
                metadata.change.emit()
                //=========================================================""")

        self.callback_start = self.callback_start_0.clone()
        self.callback_end   = self.callback_end_0.clone()
        self.range.js_on_change('start',self.callback_start )
        self.range.js_on_change('end'  ,self.callback_end )
        #-------------------------
        

    def add_start_callback(self,args,code):
        _args = dict(self.callback_start_0.args)
        _args.update(args)
        self.callback_start.update(args=_args, 
                                   code=self.callback_start_0.code + '\n' + code)

    def add_end_callback(self,args,code):
        _args = dict(self.callback_end_0.args)
        _args.update(args)
        self.callback_end.update(   args=_args, 
                                    code=self.callback_end_0.code + '\n' + code)
        


def make_bbbsignature_figure(source,beam,bb_df,colors = {'left':'coral','right':'mediumseagreen'}):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = _default_fig_height, 
                    width           = _default_fig_width,
                    title           = "Normalised losses", 
                    tools           = "box_zoom,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    toolbar_location= "right")

    # No grid
    # fig.grid.visible = False
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]
    # fig.tags[0]['BoxZoomTool'].update(dimensions = 'width')
    # fig.tags[0]['PanTool'].update(dimensions = 'width')
    fig.tags[0]['HoverTool'].update(tooltips = [('Bunch slot','$x{0}'),(f'Value', '$y')])
    #=====================================

    
    c_left  = fig.circle(x='Bunch', y='data_l'  , color=colors['left']  ,name=f"{beam.name}",legend_label= 'Cursor 1'   ,source=source[beam.name],fill_alpha=0.5,size=6,line_width=1,line_color=colors['left'])
    c_avg   = fig.circle(x='Bunch', y='data_avg', color=colors['avg']   ,name=f"{beam.name}",legend_label= 'Average'    ,source=source[beam.name],fill_alpha=0.5,size=6,line_width=1,line_color=colors['avg']) 
    c_right = fig.circle(x='Bunch', y='data_r'  , color=colors['right'] ,name=f"{beam.name}",legend_label= 'Cursor 2'   ,source=source[beam.name],fill_alpha=0.5,size=6,line_width=1,line_color=colors['right'])


    fig.xaxis.axis_label = "Bunch slot"
    fig.yaxis.axis_label = "Normalised losses: losses/(sigma*lumi)"
    fig.x_range          = bkmod.Range1d(-10, len(b_slots)+10)
    fig.y_range          = bkmod.Range1d(-0.2, 3)


    # Adding BBB Signature
    # Finding schedule info
    schedule_dict = {}
    for exp,ip in zip(['ATLAS','CMS','ALICE','LHCB'],[1,5,2,8]):
        schedule_dict[f'LR IP{ip} ({exp})'] = bb_df[f'# of LR in {exp}'].values.astype(float)
        schedule_dict[f'HO IP{ip} ({exp})'] = bb_df[f'HO partner in {exp}'].apply(lambda line: (~np.isnan(line)).astype(float)).values
    schedule_df = pd.DataFrame(schedule_dict,index = bb_df.index)
    schedule_df.insert(0,'Bunch',bb_df.index)
    schedule_df.insert(1,'Default',0.0)
    schedule_df.insert(1,'Active',0.0)
    #schedule_df

    # Creating Checkbox list
    source_sched = bkmod.ColumnDataSource(schedule_df)
    checklist    = ['Clear all'] + sorted(list(schedule_df.columns[3:]))
    widget    = bkmod.CheckboxGroup(labels=checklist, active=[])

    slider_amp    = bkmod.Slider(start=0.01, end=2, value=1, step=0.01,title='Amplitude')
    slider_offset = bkmod.Slider(start=-5, end=5, value=0, step=0.05, title="Offset")




    # checkbox_group = CheckboxGroup(labels=LABELS, active=[0, 1])
    callback = bkmod.callbacks.CustomJS(args = dict(source        = source_sched,
                                                    widget        = widget,
                                                    checklist     = checklist,
                                                    slider_amp    = slider_amp,
                                                    slider_offset = slider_offset),
                                                code = """
        //====================================================
        //--------------------------------
        function addvector(a,b){
            return a.map((e,i) => e + b[i]);
        }
        //--------------------------------



        if (widget.active.includes(0)){
            widget.active = [];
            slider_amp.value = 1;
            slider_offset.value = 0;
        }

        
        var active_col = source.data['Default']
        for (const _col of widget.active) {
            active_col = addvector(active_col,source.data[checklist[_col]]);
        }



        
        const max_value = Math.max.apply(Math, active_col);
        active_col = active_col.map((e,i) => e/max_value*slider_amp.value + slider_offset.value);





        source.data['Active'] = active_col;
        source.change.emit();
        slider_amp.change.emit();
        slider_offset.change.emit();
        //====================================================""")

    widget.js_on_change('active',callback)
    slider_amp.js_on_change('value',callback)
    slider_offset.js_on_change('value',callback)


    # Creating new axis for bb_signature
    ax,axis_name = new_axis(fig,axis_name='bb_signature')
    ax.axis_label = "Beam-Beam Interactions"
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1.05,min_interval=1.05)


    # Adding schedule
    bbb_count  = fig.circle(x='Bunch', y='Active'  , color='black' ,y_range_name='bb_signature',legend_label='BB Schedule' ,source=source_sched,fill_alpha=0.4,size=6,line_width=1,line_color='black')
    bbb_count.visible = False

    # Legend Options
    #=====================================
    fig.legend.location     = "top_left"
    fig.legend.click_policy = "hide"
    fig.legend.title        = f"Beam {beam.name[-1]}"
    #=====================================


    return fig,bklay.column(widget,slider_amp,slider_offset)



def make_intensity_figure(source,data_I,beam,color):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = _default_fig_height//2, 
                    width           = _default_fig_width,
                    title           = "Bunch Intensity", 
                    tools           = "box_zoom,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    toolbar_location= "right")

    # No grid
    fig.grid.visible = False
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]
    # fig.tags[0]['BoxSelectTool'].update(dimensions='width',persistent=False,icon=Path('WireDAQ/Icons/selector_icon.png'))
    # fig.tags[0]['BoxZoomTool'].update(dimensions = 'width')
    # fig.tags[0]['PanTool'].update(dimensions = 'width')
    fig.tags[0]['HoverTool'].update(tooltips = [(f'Beam', '$name'),('Bunch slot','$x{0}')])
    #=====================================


    vbars = fig.vbar(x='Bunch', top='Intensity', width=0.8,color=color,name=f"{beam.name}",source=source[beam.name]) 

    # vbars.selection_glyph = bkmod.glyphs.VBar(line_color='black',fill_color=color)
    vbars.nonselection_glyph = bkmod.glyphs.VBar(fill_color=color,fill_alpha=0.3,line_color=None)

    # Vertical line
    # l1 = bkmod.Span(location=0, dimension='height', line_color='black',line_alpha=0.5)
    # l2 = bkmod.Span(location=len(b_slots), dimension='height', line_color='black',line_alpha=0.5)
    # fig.renderers.extend([l1,l2])
    fig.xaxis.axis_label = "Bunch slot"
    fig.yaxis.axis_label = "Intensity [1e11 p+]"
    # fig.yaxis.ticker     = [0,1]
    fig.x_range          = bkmod.Range1d(-10, len(b_slots)+10)
    # fig.yaxis.formatter  = bkmod.NumeralTickFormatter(format="0.0")
    fig.y_range          = bkmod.Range1d(0, 1.05*data_I[beam.name].apply(lambda line: np.max(line)).max())

    return fig

def make_emittance_figure(source,data_E,plane,beam,color):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = _default_fig_height//2, 
                    width           = _default_fig_width,
                    title           = f"Bunch Emittance {plane}", 
                    tools           = "box_zoom,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    toolbar_location= "right")

    # No grid
    fig.grid.visible = False
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]
    # fig.tags[0]['BoxSelectTool'].update(dimensions='width',persistent=False,icon=Path('WireDAQ/Icons/selector_icon.png'))
    # fig.tags[0]['BoxZoomTool'].update(dimensions = 'width')
    # fig.tags[0]['PanTool'].update(dimensions = 'width')
    fig.tags[0]['HoverTool'].update(tooltips = [(f'Beam', '$name'),('Bunch slot','$x{0}')])
    #=====================================


    vbars = fig.vbar(x='Bunch', top=f'Emitt_{plane}', width=0.8,color=color,name=f"{beam.name}",source=source[beam.name]) 

    # vbars.selection_glyph = bkmod.glyphs.VBar(line_color='black',fill_color=color)
    vbars.nonselection_glyph = bkmod.glyphs.VBar(fill_color=color,fill_alpha=0.3,line_color=None)

    # Vertical line
    # l1 = bkmod.Span(location=0, dimension='height', line_color='black',line_alpha=0.5)
    # l2 = bkmod.Span(location=len(b_slots), dimension='height', line_color='black',line_alpha=0.5)
    # fig.renderers.extend([l1,l2])
    fig.xaxis.axis_label = "Bunch slot"
    fig.yaxis.axis_label = "Emittance [um]"
    # fig.yaxis.ticker     = [0,1]
    fig.x_range          = bkmod.Range1d(-10, len(b_slots)+10)
    # fig.yaxis.formatter  = bkmod.NumeralTickFormatter(format="0.0")
    fig.y_range          = bkmod.Range1d(0, 1.05*data_E[beam.name].apply(lambda line: np.max(line)).max())

    return fig

def export_HTML(LAYOUT,filename,tabname):

    bk.output_file(filename=filename, title=tabname)
    bk.save(LAYOUT)

    print(f'Saved {tabname}:{filename}')




# To call the script directly
if __name__ == '__main__':
    import argparse
    # Adding command line parser
    aparser = argparse.ArgumentParser()
    aparser.add_argument("FILL", type=int,                              help = "Fill number to analyse")
    aparser.add_argument("-p", "--path",                                help = "Location of parquet files"  ,default = _default_path)
    aparser.add_argument("-d", "--device",choices=['DBLM', 'BCTF'],     help = "Location of parquet files"  ,default = _default_device)
    aparser.add_argument("-e", "--export",                              help = "Location to export HTML"    ,default = _default_out)
    args = aparser.parse_args()
    
    assert args.FILL>8000, 'Invalid fill number'
    assert Path(args.path).exists(), 'Invalid data path'
    assert Path(args.export).exists(), 'Invalid export path'
    
    print(40*'*')
    print(f' bbbsignature_HTML on F{args.FILL}\n | Device: \t{args.device}\n | Data path: \t{args.path}\n | Export to: \t{args.export}/FILL{args.FILL}.html')
    print(40*'*')

    Efficiency_to_HTML(args.FILL, HTML_name= args.export + f'/FILL{args.FILL}.html',device=args.device,data_path=args.path)
