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
_default_out      = f'/eos/user/p/phbelang/www/Monitoring_efficiency/{_default_device}'


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
    slider_dt     = 5*60
    default_ts    = database['Timestamp'].iloc[len(database)//2]
    source,data_I,data_E_H,data_E_V = make_efficiency_source(FILL,database,bb_df_list=[bb_df_b1,bb_df_b2],slider_dt=slider_dt,default_ts = default_ts)
    #-------------------------------------

    # Overview plot
    #-------------------------------------
    BOKEH_FIGS['Overview'] = make_overview_figure(FILL,database)

    # Add slider tool
    #-------------------------------------
    time_slider = Slider(BOKEH_FIGS['Overview'],x=default_ts,w=pd.Timedelta(seconds=slider_dt),fill_color='black',fill_alpha=0.2,line_alpha=1)
    time_slider.add_callback(args = dict(   sourceb1= source['B1'],
                                            sourceb2= source['B2'],
                                            ts_list = list(data_I['B1'].index),
                                            bb_I_b1 = data_I['B1'],
                                            bb_I_b2 = data_I['B2'],
                                            bb_EH_b1 = data_E_H['B1'],
                                            bb_EH_b2 = data_E_H['B2'],
                                            bb_EV_b1 = data_E_V['B1'],
                                            bb_EV_b2 = data_E_V['B2']),
            code =  """
                    const _s     = metadata.data['x'][0]-metadata.data['w'][0];
                    const _s_idx = ts_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_s)));
                    sourceb1.data.Intensity = bb_I_b1[_s_idx];
                    sourceb2.data.Intensity = bb_I_b2[_s_idx];
                    sourceb1.data.Emitt_H = bb_EH_b1[_s_idx];
                    sourceb2.data.Emitt_H = bb_EH_b2[_s_idx];
                    sourceb1.data.Emitt_V = bb_EV_b1[_s_idx];
                    sourceb2.data.Emitt_V = bb_EV_b2[_s_idx];

                    sourceb1.change.emit();
                    sourceb2.change.emit();
                    """)
    #-------------------------------------

    # Efficiency plot
    #-------------------------------------
    for beam,bb_df,color in zip(beams,[bb_df_b1,bb_df_b2],['royalblue','firebrick']):
        BOKEH_FIGS[f'Efficiency {beam.name}'] = make_efficiency_figure(database,source,beam,color)#make_efficiency_figure(FILL,database,source,data_I,beam,bb_df,color,slider_source,ghost_range,dt_I = dt_I,default_ts = default_ts)
        # Adding slider indicator
        #-------------
        time_slider.add_renderer(BOKEH_FIGS[f'Efficiency {beam.name}'],fill_color='black',fill_alpha=0.2,line_alpha=0)
        #-------------


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


    # linking efficiency-overview
    #---------------------------------
    # BOKEH_FIGS['Overview'].x_range.update(bounds = 'auto')#max_interval=BOKEH_FIGS['Overview'].x_range.end-BOKEH_FIGS['Overview'].x_range.start)

    BOKEH_FIGS[f'Efficiency B1'].x_range = BOKEH_FIGS['Overview'].x_range


    # linking both efficiencies and making panel
    #---------------------------------
    BOKEH_FIGS[f'Efficiency B2'].x_range = BOKEH_FIGS[f'Efficiency B1'].x_range
    BOKEH_FIGS[f'Efficiency B2'].y_range = BOKEH_FIGS[f'Efficiency B1'].y_range

    

    BOKEH_FIGS[f'Efficiency B1B2'] = bkmod.Tabs(tabs=[  bkmod.TabPanel(child=BOKEH_FIGS[f'Efficiency B1'], title="Beam 1"), 
                                                        bkmod.TabPanel(child=BOKEH_FIGS[f'Efficiency B2'], title="Beam 2")])

    # Linking both intensities and making grid
    #---------------------------------
    BOKEH_FIGS[f'Intensity B2'].x_range = BOKEH_FIGS[f'Intensity B1'].x_range
    BOKEH_FIGS[f'Intensity B2'].y_range = BOKEH_FIGS[f'Intensity B1'].y_range

    for beam in beams:
        for plane in ['H','V']:
            BOKEH_FIGS[f'Emitt_{plane} {beam.name}'].x_range = BOKEH_FIGS[f'Intensity B1'].x_range
            BOKEH_FIGS[f'Emitt_{plane} {beam.name}'].y_range = BOKEH_FIGS[f'Emitt_H B1'].y_range


    # BOKEH_FIGS['Intensity B1B2'] =  bklay.gridplot([[BOKEH_FIGS['Intensity B1']], [BOKEH_FIGS['Intensity B2']]],toolbar_location='right')
        # Making tabs
    BOKEH_FIGS[f'Int. and Emitt'] = bkmod.Tabs(tabs=[  bkmod.TabPanel(child=bklay.column(BOKEH_FIGS['Intensity B1'],BOKEH_FIGS['Intensity B2']), title="Intensity"), 
                                                       bkmod.TabPanel(child=bklay.column(BOKEH_FIGS['Emitt_H B1'],BOKEH_FIGS['Emitt_H B2'])    , title="Emittance H"),
                                                       bkmod.TabPanel(child=bklay.column(BOKEH_FIGS['Emitt_V B1'],BOKEH_FIGS['Emitt_V B2'])    , title="Emittance V")])


    
    # Making final layout
    #---------------------------------
    HTML_LAYOUT = bklay.column(bklay.gridplot([[BOKEH_FIGS['Overview']], 
                                               [BOKEH_FIGS[f'Efficiency B1B2']]],toolbar_location='right'),
                                bklay.gridplot([[BOKEH_FIGS[f'Int. and Emitt']]],toolbar_location='right')
                                               )
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

    # data_path = _default_path
    # device    = _default_device
    # FILL = 8850

    # Fixing data path
    raw_data    = data_path + '/rawdata/'
    # device_data = data_path + f'/efficiency_data/{device}/'
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
    n_colliding   = len(bb_df_b1['HO partner in ATLAS'].dropna())
    avg_per_bunch = df_lumi_tot['bb_LUMI_TOT'].apply(lambda line:np.sum(line)/n_colliding)
    noise_level   = 0.00115

    ROI_idx = avg_per_bunch[avg_per_bunch>5*noise_level].index[[0,-1]]
    df_eff  = df_eff.loc[ROI_idx[0]:ROI_idx[1]]
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

    # Intensity
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


def make_efficiency_source(FILL,database,bb_df_list=None,slider_dt = 120,default_ts = None):
    # Creating shared source for plots
    #=====================================================================
    source = {}
    data_I = {}
    data_E_H= {}
    data_E_V= {}
    for beam,bb_df,color in zip(beams,bb_df_list,['royalblue','firebrick']):
        
        
        # Extracting efficiency
        observable   = f'{beam.name}:eta'
        _times,_data = database.set_index('Timestamp')[observable].dropna().to_2D()
        _times       = _times.to_list()

        # Computing average intensity
        _data_I           = database.set_index('Timestamp')[beam.bb_Intensity].dropna()
        data_I[beam.name] = _data_I.groupby(pd.Grouper(freq=f'{slider_dt}s')).mean().apply(lambda line: line[bb_df.index]/1e11)


        _data_           = database.set_index('Timestamp')[beam.bb_Emittance_H].dropna()
        data_E_H[beam.name] = _data_.groupby(pd.Grouper(freq=f'{slider_dt}s')).mean().apply(lambda line: line[bb_df.index])

        _data_           = database.set_index('Timestamp')[beam.bb_Emittance_V].dropna()
        data_E_V[beam.name] = _data_.groupby(pd.Grouper(freq=f'{slider_dt}s')).mean().apply(lambda line: line[bb_df.index])


        # Compiling data to source
        _source_df   = pd.DataFrame({'Bunch'   :bb_df.index,
                                    'Intensity':data_I[beam.name][default_ts:][0],
                                    'Emitt_H'   :data_E_H[beam.name][default_ts:][0],
                                    'Emitt_V'   :data_E_V[beam.name][default_ts:][0],
                                    'Timestamp':[_times       for idx in bb_df.index],
                                    'eta'      :[_data[:,idx] for idx in bb_df.index]})
        
        source[beam.name] = bkmod.ColumnDataSource(_source_df)
        source[beam.name].selected.update(indices=_source_df.index[-10:])
    #=====================================================================

    return source,data_I,data_E_H,data_E_V



class Slider():
    def __init__(self,_fig,x,w,**kwargs):
        
        # Icon file
        self.icon     = Path('WireDAQ/Icons/slider_icon.png')

        # Creating new axis from 0-1 on the figure
        ax,axis_name = new_axis(_fig,axis_name='_slider')
        _fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1)
        kwargs.update({'y_range_name':axis_name})

        # Metadata
        w_factor = 3
        self.metadata  = bkmod.ColumnDataSource({'x': [x], 'w':[w],'w_drag':[w_factor*w],'w_factor':[w_factor] ,'yt':[2],'yb':[-1],'ym':[0.5]})

        # Range object
        _s_ts = x-w_factor*w/2
        _e_ts = x+w_factor*w/2
        self.range = bkmod.Range1d(_s_ts,_e_ts)


        # Add tool to figure
        self.RangeTool = bkmod.RangeTool(   x_range      = self.range,
                                            icon         = self.icon)
        
        self.RangeTool.overlay.update(      visible      = False,
                                            fill_color   = None,
                                            fill_alpha   = 0,
                                            line_color   = None)
        _fig.add_tools(self.RangeTool)


        # Add visible slider itself
        self.renderer = _fig.rect(x="x", y="ym", width="w", height="yt", source=self.metadata,**kwargs)

        # Muting for overlay tool
        self.renderer.muted = True
        if 'y_range_name' in kwargs.keys():
            kwargs.pop('y_range_name');
        self.renderer.muted_glyph.update(**kwargs)

        # Default callback to link the range with rendered object
        #-------------------------
        self.callback_start = bkmod.callbacks.CustomJS(args=dict(metadata = self.metadata), 
                                                                            code="""
                //=========================================================
                cb_obj.end         = cb_obj.start+metadata.data['w_drag'][0];
                metadata.data['x'] = [(cb_obj.start + cb_obj.end)/2];
                metadata.change.emit()
                //=========================================================""")

        self.callback_end = bkmod.callbacks.CustomJS(args=dict(metadata = self.metadata), 
                                                                            code="""
                //=========================================================
                cb_obj.end         = cb_obj.start+metadata.data['w_drag'][0];
                metadata.data['x'] = [(cb_obj.start + cb_obj.end)/2];
                metadata.change.emit()
                //=========================================================""")

        self.range.js_on_change('start',self.callback_start )
        self.range.js_on_change('end'  ,self.callback_end )
        #-------------------------
        
    def add_renderer(self,_fig,**kwargs):
        # Creating new axis from 0-1 on the figure
        _,axis_name = new_axis(_fig,axis_name='_slider_indicator')
        _fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1)
        kwargs.update({'y_range_name':axis_name})

        _rend = _fig.rect(x="x", y="ym", width="w", height="yt", source=self.metadata,**kwargs)
        # Muting for overlay tool
        _rend.muted = True
        if 'y_range_name' in kwargs.keys():
            kwargs.pop('y_range_name');
        _rend.muted_glyph.update(**kwargs)
        return _rend
    
    def update(self,**kwargs):
        if 'y_range_name' in kwargs.keys():
            kwargs.pop('y_range_name');
        self.renderer.glyph.update(**kwargs)
        self.renderer.muted_glyph.update(**kwargs)

    def add_callback(self,args,code):
        self.callback_start.update(args=dict(metadata = self.metadata,**args), 
                                                                            code="""
                //=========================================================
                cb_obj.end         = cb_obj.start+metadata.data['w_drag'][0];
                metadata.data['x'] = [(cb_obj.start + cb_obj.end)/2];
                metadata.change.emit()
                //=========================================================\n""" + code)


def make_efficiency_figure(database,source,beam,color):


    # Creating Figure
    #=====================================
    fig = bk.figure(height          = _default_fig_height, 
                    width           = _default_fig_width,
                    title           = "Efficiency",
                    x_axis_type     = "datetime", 
                    tools           = "pan,box_zoom,reset,save,hover",
                    active_drag     = "box_zoom",
                    x_range         =  bkmod.Range1d(start=source[beam.name].data['Timestamp'][0][0],end=source[beam.name].data['Timestamp'][0][-1],bounds='auto'))
    
    # Datetime format
    fig.xaxis.formatter= bkmod.DatetimeTickFormatter(hourmin = '%H:%M',hours='%H:%M',days='%H:%M',months='%H:%M',years='%H:%M')
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]

    # fig.tags[0]['PanTool'].update(dimensions = 'width')
    fig.tags[0]['HoverTool'].update(tooltips=[('Bunch', '@Bunch'),('Time (H:M)','$x{%H:%M}'),('Value','$y')],formatters={ "$x": "datetime"},muted_policy='ignore')
    #=====================================

    # Plotting efficiency
    #--------------------
    # source[beam.name].selected.update(indices=source[beam.name].data.index[-10:])
    mlines = fig.multi_line(xs='Timestamp', ys='eta',source=source[beam.name],color=color)

    # Updating nonselection glyph
    mlines.nonselection_glyph = bkmod.glyphs.MultiLine(line_color=None, line_width=0)

    # Axis info
    fig.y_range          = bkmod.Range1d(0, 1.05)
    fig.yaxis.axis_label = "Burn-off efficiency"
    fig.xaxis.axis_label = f"Local Time, {database['Timestamp'].iloc[0].strftime('%Y-%m-%d')}"
    #--------------------

    return fig



def make_intensity_figure(source,data_I,beam,color):

    # Creating Figure
    #=====================================
    fig = bk.figure(output_backend  = "webgl",
                    height          = _default_fig_height//2, 
                    width           = _default_fig_width,
                    title           = "Bunch Intensity", 
                    tools           = "box_zoom,box_select,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    toolbar_location= "right")

    # No grid
    fig.grid.visible = False
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]
    fig.tags[0]['BoxSelectTool'].update(dimensions='width',persistent=False,icon=Path('WireDAQ/Icons/selector_icon.png'))
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
                    tools           = "box_zoom,box_select,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    toolbar_location= "right")

    # No grid
    fig.grid.visible = False
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]
    fig.tags[0]['BoxSelectTool'].update(dimensions='width',persistent=False,icon=Path('WireDAQ/Icons/selector_icon.png'))
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
    print(f' efficiency_HTML on F{args.FILL}\n | Device: \t{args.device}\n | Data path: \t{args.path}\n | Export to: \t{args.export}/FILL{args.FILL}.html')
    print(40*'*')

    Efficiency_to_HTML(args.FILL, HTML_name= args.export + f'/FILL{args.FILL}.html',device=args.device,data_path=args.path)
