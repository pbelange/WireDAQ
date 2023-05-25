import numpy as np
import pandas as pd
from pathlib import Path

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
_default_device = 'DBLM'

_default_import = 'local'
_default_path   = '/home/lumimod/work/run/data/2023/rawdata/'
_default_out    = '/eos/user/p/phbelang/www/Monitoring_BBCW/'
#------------------------------------------------



def Efficiency_to_HTML(FILL, device=_default_device, HTML_name=None, import_from = _default_import,local_path=_default_path):
    if HTML_name is None:
        HTML_name   = _default_out + f'FILL{FILL}.html'

    print('\n' + 40*'=')
    print('IMPORTING DATA')
    database,BCTF_efficiency,DBLM_efficiency,bb_df_b1,bb_df_b2 = run_analysis(FILL,import_from =import_from,local_path=local_path)
    print(40*'=')

    # Cropping the filling for testing
    bb_df_b1 = bb_df_b1.loc[1070:2050]
    bb_df_b2 = bb_df_b2.loc[1070:2050]

    print('\n' + 40*'=')
    print('CREATING FIGURES')
    # Creating figures
    #=====================================
    BOKEH_FIGS = {}
    BOKEH_FIGS['Overview'] = make_overview_figure(FILL,database)

    # Efficiency
    # Choosing efficiency data
    efficiency_df = {'DBLM':DBLM_efficiency,'BCTF':BCTF_efficiency}[device.upper()]

    dt_I          = 120
    default_ts    = database['Timestamp'].iloc[len(database)//2]
    source,data_I = make_efficiency_source(FILL,database,efficiency_df,bb_df_list=[bb_df_b1,bb_df_b2],dt_I = dt_I,default_ts = default_ts)

    slider_source,ghost_range = make_plot_slider(FILL,database,source,data_I,dt_I = dt_I,default_ts = default_ts)

    for beam,bb_df,color in zip(beams,[bb_df_b1,bb_df_b2],['royalblue','firebrick']):
        BOKEH_FIGS[f'Efficiency {beam.name}'] = make_efficiency_figure(FILL,database,source,data_I,beam,bb_df,color,slider_source,ghost_range,dt_I = dt_I,default_ts = default_ts)

    # Intensity
    for beam,bb_df,color in zip(beams,[bb_df_b1,bb_df_b2],['royalblue','firebrick']):
        BOKEH_FIGS[f'Intensity {beam.name}'] = make_intensity_figure(FILL,database,source,data_I,beam,bb_df,color)
                                          
    #=====================================
    print(40*'=')




    print('\n' + 40*'=')
    print('Exporting to HTML')
    # LAYOUT
    #=====================================

    # linking efficiency-overview
    #---------------------------------
    BOKEH_FIGS[f'Efficiency B2'].x_range = BOKEH_FIGS['Overview'].x_range


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

    BOKEH_FIGS['Intensity B1B2'] =  bklay.gridplot([[BOKEH_FIGS['Intensity B1']], [BOKEH_FIGS['Intensity B2']]],toolbar_location='right')
    
    
    # Making final layout
    #---------------------------------
    HTML_LAYOUT = bklay.column(BOKEH_FIGS['Overview'],BOKEH_FIGS[f'Efficiency B1B2'] ,BOKEH_FIGS['Intensity B1B2'])

    #=====================================

    # Exporting to HTML
    #=====================================
    export_HTML(HTML_LAYOUT,HTML_name,f'Fill {FILL}')
    print(40*'=')




def run_analysis(FILL,import_from = _default_import,local_path=_default_path):
    # Variables to be used
    #============================================
    LHC_vars   = LHC._getVarList(subset=[   'bb_Luminosity',
                                            'Luminosity',
                                            'Xing',
                                            'Fill',
                                            'Bmode',
                                            'betastar'])
    LHC_vars.remove(LHC.bb_Luminosity['LHCB'])
    LHC_vars.remove(LHC.bb_Luminosity['ALICE'])

    beams_vars = sum([beam._getVarList(subset=[ 'bb_Intensity',
                                                'bb_Intensity_B',
                                                'Intensity',
                                                'Intensity_B',
                                                'Nb',
                                                'dBLM_Amp']) for beam in beams],[])


    wires_vars = sum([wire._getVarList(subset=[ 'I']) for wire in wires['B2']],[])
    variables = sorted(LHC_vars+beams_vars+wires_vars)
    #============================================


    # Loading the data
    #============================================
    if import_from.lower() == 'nxcals':
        database      = parser.from_NXCALS(fill=FILL,variables = variables)
    elif import_from.lower() == 'local':
        database      = parser.from_EOS(fill=FILL,variables = variables,DATA_PATH = local_path)
    #============================================

    # Extracting filling pattern
    #============================================
    bb_df_b1,bb_df_b2 = parser.getFillingPattern(database)
    #============================================

    # Computing intensity avg (every 20 seconds) with calibration BCT_A+BCT_B, aligned with ATLAS
    #============================================
    BCT_avg  = eff.calibrate_BCT(database,dt=20,calibration_ts = None,reference = LHC['bb_Luminosity']['ATLAS'])
    database = pd.concat([database,BCT_avg])
    database = database.sort_index()
    #============================================

    # Computing Lumi tot (ATLAS + CMS bunch by bunch)
    #============================================
    Lumi_tot = eff.computeLumiTot(database,experiments = ['ATLAS','CMS'])
    database = pd.concat([database,Lumi_tot])
    database = database.sort_index()
    #============================================


    # BCTF
    #==============================================================
    # Computing efficiency 
    BCTF_efficiency = {}
    for beam in beams:
        BCTF_efficiency[beam.name]  = eff.compute_BCTF_efficiency(database,beam,dt= 60,smooth_derivative=False)
    #==============================================================

    # DBLM
    #==============================================================
    # Computing efficiency 
    DBLM_efficiency = {}
    for beam in beams:
        # DBLM_efficiency[beam.name]  = eff.compute_dBLM_efficiency(database,beam,dt = 10,baseline = None,calib_ROI=None,filled_b_slots=None)
        DBLM_efficiency[beam.name]  = eff.compute_dBLM_efficiency(database,beam,dt = 30,baseline = None,calib_ROI=None,filled_b_slots=None)
    #==============================================================

    return database,BCTF_efficiency,DBLM_efficiency,bb_df_b1,bb_df_b2

def make_overview_figure(FILL,database):
    # Creating Figure
    #=====================================
    fig = bk.figure(height          = 500, 
                    width           = 1500,
                    title           = "Overview" + f' FILL {FILL:d}  ({database["Timestamp"].iloc[0].strftime("%Y-%m-%d")})', 
                    x_axis_type     = "datetime",
                    tools           = "pan,wheel_zoom,box_zoom,reset,save",
                    active_drag     = "box_zoom")
    fig.xaxis.formatter= bkmod.DatetimeTickFormatter(hourmin = '%H:%M',hours='%H:%M',days='%H:%M',months='%H:%M',years='%H:%M')
    fig.add_tools(bkmod.HoverTool(
        tooltips=[('Variable', '$name'),('Time (H:M)','$x{%H:%M}'),('Value','$y')],
        formatters={ "$x": "datetime"}))
    #=====================================

    # New axis function
    #=====================================
    def new_axis(fig,axis_name,side='left'):
        fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1)
        _ax = bkmod.LinearAxis(y_range_name=axis_name)
        fig.add_layout(_ax,side)

        return _ax,axis_name
    #=====================================


    # Plotting Intensity
    #--------------------
    for beam,color in zip(beams,['blue','red']):
        data = database.set_index('Timestamp')[beam.Intensity].dropna()
        
        fig.line(data.index,data,color=color,alpha=0.8,legend_label=beam.name,name=f'Intensity {beam.name}')

    fig.yaxis.axis_label = "Intensity [1e14 p+]"
    fig.xaxis.axis_label = f"Local Time, {database['Timestamp'].iloc[0].strftime('%Y-%m-%d')}"
    #--------------------

    # Plotting Luminosity
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='Luminosity',side='left')
    max_y = 0
    for loc,color in zip(['ATLAS','CMS'],['orange','green']):
        data = database.set_index('Timestamp')[LHC['bb_Luminosity'][loc]].dropna()
        data = data.apply(lambda line: np.sum(line))
        
        fig.line(data.index,data,color=color,alpha=0.8,legend_label=loc,name=f'Luminosity {loc}',y_range_name=axis_name)

        max_y  = np.max((max_y,np.max(data)))

    fig.extra_y_ranges[axis_name] = bkmod.Range1d(-0.05*max_y,1.05*max_y)
    ax.axis_label = 'Luminosity [Hz/ub]'
    #--------------------


    # Plotting xing
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='xing',side='right')


    data = database.set_index('Timestamp')[LHC.Xing['IP5']].dropna()
    fig.line(data.index,data,color='indianred',alpha=0.8,legend_label="theta/2",name="theta/2",y_range_name=axis_name)

        
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(110,180)
    ax.axis_label = r"Half-crossing angle [urad]"
    #--------------------


    # Plotting beta star
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='beta',side='right')


    data = database.set_index('Timestamp')[LHC.betastar['IP5']].dropna()
    data[(data>90)|(data<0)]=np.nan
    fig.line(data.index,data,color='purple',alpha=0.8,legend_label="beta*",name="beta*",y_range_name=axis_name)

        
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,90)
    ax.axis_label = r"Beta star [cm]"
    #--------------------




    # Plotting wire current
    #--------------------
    ax,axis_name = new_axis(fig,axis_name='wire',side='right')

    for wire in wires['B2'] :
        data = database.set_index('Timestamp')[wire.I].dropna()

        fig.line(data.index,data,color='teal',alpha=0.8,legend_label="BBCW",name="BBCW",y_range_name=axis_name)

    max_y = 2000
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(-0.05*max_y,1.05*max_y)
    ax.axis_label = r"BBCW Current [A]"
    #--------------------

    # Legend Options
    #=====================================
    fig.legend.location     = "top_left"
    fig.legend.click_policy = "hide"
    #=====================================

    return fig


def make_efficiency_source(FILL,database,efficiency_df,bb_df_list=None,dt_I = 120,default_ts = None):
    # Creating shared source for plots
    #=====================================================================
    source = {}
    data_I = {}
    for beam,bb_df,color in zip(beams,bb_df_list,['royalblue','firebrick']):
        
        
        # Extracting efficiency
        observable   = 'eta'
        _times,_data = efficiency_df[beam.name].set_index('Timestamp')[observable].dropna().to_2D()
        _times       = _times.to_list()

        # Computing average intensity
        _data_I           = database.set_index('Timestamp')[beam.bb_Intensity].dropna()
        data_I[beam.name] = _data_I.groupby(pd.Grouper(freq=f'{dt_I}s')).mean().apply(lambda line: line[bb_df.index]/1e11)

        # Compiling data to source
        _source_df   = pd.DataFrame({'Bunch'   :bb_df.index,
                                    'Intensity':data_I[beam.name][default_ts:][0],
                                    'Timestamp':[_times       for idx in bb_df.index],
                                    'eta'      :[_data[:,idx] for idx in bb_df.index]})
        
        source[beam.name] = bkmod.ColumnDataSource(_source_df)
        source[beam.name].selected.update(indices=_source_df.index[-10:])
    #=====================================================================

    return source,data_I


def make_plot_slider(FILL,database,source,data_I,dt_I = 120,default_ts = None):
    # Creating plot slider
    #=====================================================================

    slider_dt = pd.Timedelta(seconds=dt_I)
    ghost_dt  = pd.Timedelta(minutes=15)

    start_ts  = default_ts-ghost_dt/2
    stop_ts   = default_ts+ghost_dt/2
    middle_ts = default_ts

    ghost_range   = bkmod.Range1d(start_ts,stop_ts,reset_start = start_ts,reset_end=stop_ts)
    slider_source = bkmod.ColumnDataSource(data=dict(x=[start_ts + ghost_dt/2,start_ts + ghost_dt/2], ym=[-10, 10]))

    ghost_range.js_on_change('start', bkmod.callbacks.CustomJS(args=dict(   slider  = ghost_range ,
                                                                            dt      = ghost_dt,
                                                                            s2      = slider_source,
                                                                            sourceb1= source['B1'],
                                                                            sourceb2= source['B2'],
                                                                            ts_list = list(data_I['B1'].index),
                                                                            bb_I_b1 = data_I['B1'],
                                                                            bb_I_b2 = data_I['B2']), 
                                                                            code="""
        //=========================================================
        slider.end = cb_obj.start+dt;
        s2.data    = {x:[slider.start+dt/2,slider.start+dt/2],ym:[-10, 10]};


        const _s     = slider.start+dt/2;
        const _s_idx = ts_list.findIndex( x => Date.parse(new Date(x)) >= Date.parse(new Date(_s)));
        sourceb1.data.Intensity = bb_I_b1[_s_idx];
        sourceb2.data.Intensity = bb_I_b2[_s_idx];
        sourceb1.change.emit();
        sourceb2.change.emit();
        //========================================================="""))


    ghost_range.js_on_change('end', bkmod.callbacks.CustomJS(args=dict(     slider  = ghost_range ,
                                                                            dt      = ghost_dt,
                                                                            s2      = slider_source),
                                                                            code    ="""
        //=========================================================
        slider.end = cb_obj.start+dt;
        s2.data    = {x:[slider.start+dt/2,slider.start+dt/2],ym:[-10, 10]};
        //========================================================="""))
    #=====================================================================

    return slider_source,ghost_range



def make_efficiency_figure(FILL,database,source,data_I,beam,bb_df,color,slider_source,ghost_range,dt_I = 120,default_ts = None):

    slider_dt = pd.Timedelta(seconds=dt_I)
    ghost_dt  = pd.Timedelta(minutes=15)

    start_ts  = default_ts-ghost_dt/2
    stop_ts   = default_ts+ghost_dt/2
    middle_ts = default_ts


    # Creating Figure
    #=====================================
    fig = bk.figure(height          = 400, 
                    width           = 1500,
                    title           = "Efficiency",
                    x_axis_type     = "datetime", 
                    tools           = "box_zoom,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    x_range         =  bkmod.Range1d(start=source[beam.name].data['Timestamp'][0][0],end=source[beam.name].data['Timestamp'][0][-1],bounds='auto'))
    
    # Datetime format
    fig.xaxis.formatter= bkmod.DatetimeTickFormatter(hourmin = '%H:%M',hours='%H:%M',days='%H:%M',months='%H:%M',years='%H:%M')
    
    # Saving tools to tags
    fig.add_tools(bkmod.RangeTool())
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]

    fig.tags[0]['RangeTool'].update(x_range = ghost_range)
    fig.tags[0]['RangeTool'].overlay.update(visible=False,fill_color=None,fill_alpha=0,line_color=None)

    fig.tags[0]['PanTool'].update(dimensions = 'width')
    fig.tags[0]['HoverTool'].update(tooltips=[('Bunch', '@Bunch'),('Time (H:M)','$x{%H:%M}'),('Value','$y')],formatters={ "$x": "datetime"},muted_policy='ignore')
    #=====================================

    # Plotting efficiency
    #--------------------
    # source[beam.name].selected.update(indices=source[beam.name].data.index[-10:])
    mlines = fig.multi_line(xs='Timestamp', ys='eta',source=source[beam.name])

    # Updating nonselection glyph
    mlines.nonselection_glyph = bkmod.glyphs.MultiLine(line_color=None, line_width=0)

    # Axis info
    fig.y_range          = bkmod.Range1d(0, 1.05)
    #--------------------

    # Plotting slider
    #--------------------
    slider = fig.vbar(x='x', top='ym', color="navy",width=slider_dt,line_width=1,line_color='black',line_dash="2 2",line_alpha=1, alpha=0.2, source=slider_source)
    slider.muted = True
    #--------------------

    return fig



def make_intensity_figure(FILL,database,source,data_I,beam,bb_df,color):

    # Creating Figure
    #=====================================
    fig = bk.figure(height          = 200, 
                    width           = 1500,
                    title           = "Bunch Intensity", 
                    tools           = "box_zoom,box_select,pan,reset,save,hover",
                    active_drag     = "box_zoom",
                    toolbar_location= "right")

    # No grid
    fig.grid.visible = False
    
    # Saving tools to tags
    fig.tags = [{str(type(t)).split('.')[-1].split('\'')[0]:t for t in fig.tools}]
    fig.tags[0]['BoxSelectTool'].update(dimensions='width',persistent=False)
    # fig.tags[0]['BoxZoomTool'].update(dimensions = 'width')
    fig.tags[0]['PanTool'].update(dimensions = 'width')
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
    fig.yaxis.formatter  = bkmod.NumeralTickFormatter(format="0.0")
    fig.y_range          = bkmod.Range1d(0, 1.05*data_I[beam.name].apply(lambda line: np.max(line)).max())

    return fig


def export_HTML(LAYOUT,filename,tabname):

    bk.output_file(filename=filename, title=tabname)
    bk.save(LAYOUT)

    print(f'Saved {tabname}:{filename}')





# To call the script directly
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 :
        print('Invalid arguments, quitting...')
        sys.exit(0)
    elif len(sys.argv) == 2:
        inputFile = sys.argv[1]
        ufo,beam,collisionTime,inelasticCollisionss = main(inputFile,showPlots = True,saveOutput = True)
    else:
        ufo,beam,collisionTime,inelasticCollisions = main(showPlots = True,saveOutput = True)
