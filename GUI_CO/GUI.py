from matplotlib import pyplot as plt
import ipywidgets as widgets
import time
import numpy as np
import pandas as pd

import Backend.Constants as cst



def fromPkl(folder,opticsFile = 'opticsfile.1'):
    twiss = {}
    for beam in ['B1','B2']:
         twiss[beam] = pd.read_pickle(f"{folder}/opticsfile{opticsFile.split('.')[-1]}_{beam}_twiss.pkl")

    return twiss

twiss = fromPkl(folder='.',opticsFile='.1')

class myApp():
    def __init__(self,):
        
        
        # WIRES INFO FOR PHYSICS
        #==============================================
        self.E     = 450e9
        self.emitt = 2.5e-6/(1+self.E/cst.m_p_eV)
        
        # Extracted from survey
        self.B2_dist = {('ip1','x'):167.00405036118732e-3,
                        ('ip1','y'):4.2545250898809766e-3,
                        ('ip5','x'):175.60388642368642e-3,
                        ('ip5','y'):-0.0003865603020114507e-3,
                       }
        
        # SLIDERS RANGE
        self.slider_unit     = 1e-3 
        self.slider_unit_txt = '[mm]'
        self.slider_max      = 10
    
        # Making sure ioff() is on
        plt.ioff()
        
        
        #
        if False:
            showBorder = 'solid'
        else:
            showBorder = ''
        
        self.height = 400
        self.width = 600
        px = 1/plt.rcParams['figure.dpi']
        
        # BEAM
        #======================
        self.panel_beam = widgets.Output(layout=widgets.Layout(display='flex',border=showBorder,width=f'{3/5*self.width}px', height=f'{3/5*self.height}px'))
        with self.panel_beam:
            self.fig_beam, self.ax_beam = plt.subplots(constrained_layout=True, figsize=(0.9*3/5*self.width*px,0.9*3/5*self.height*px))
            self.fig_beam.canvas.resizable       = False
            self.fig_beam.canvas.footer_visible  = False
            self.fig_beam.canvas.header_visible  = False
            self.fig_beam.canvas.toolbar_visible = False
            
            
            self.coll_lines = {}
            self.coll_lines['t'], = self.ax_beam.plot([0] ,[1*self.slider_max],'o',color='C0')
            self.coll_lines['b'], = self.ax_beam.plot([0] ,[-1*self.slider_max],'o',color='C0')
            self.coll_lines['l'], = self.ax_beam.plot([-1*self.slider_max],[0],'o',color='C0')
            self.coll_lines['r'], = self.ax_beam.plot([1*self.slider_max] ,[0],'o',color='C0')
            
            
            self.ax_beam.hlines(0,-1.5*self.slider_max,1.5*self.slider_max,color='k',alpha=0.5,zorder=-1)
            self.ax_beam.axvline(0,color='k',alpha=0.5,zorder=-1)
            self.ax_beam.set_aspect('equal')#'box'
            self.ax_beam.axis([-2*self.slider_max, 1.5*self.slider_max, -1.5*self.slider_max, 1.5*self.slider_max])
            self.ax_beam.axis('off')
            
            
            display(self.fig_beam.canvas)
        #======================
        
        # Closed Orbit
        #======================
        
        self.twiss_wires = {}
        self.twiss_mu0   = {}
        self.twiss_s     = {}
        for beam in ['B1','B2']:
            for IP in ['ip1','ip5']:
                self.twiss_wires[(beam,IP)] = [name for name in twiss[beam].index if ('bbcw' in name)&((f'{IP[-1]}.u' in name)|(f'{IP[-1]}.d' in name))]
                self.twiss_s[(beam,IP)]     = twiss[beam].loc[self.twiss_wires[(beam,IP)],'s'].mean()
                for plane in ['x','y']:
                    self.twiss_mu0[(beam,IP,plane)]  = twiss[beam].loc[self.twiss_wires[(beam,IP)],f'mu{plane}'].mean()
                
                
        self.panel_CO = widgets.Output(layout=widgets.Layout(display='flex',border=showBorder,width=f'{self.width}px', height=f'{1.1*self.height}px'))
        with self.panel_CO:
            self.fig_CO, axes = plt.subplots(2,2, figsize=(1*self.width*px,0.9*1.1*self.height*px))
            self.ax_CO = {('B1','ip1'):axes[0,0],('B1','ip5'):axes[0,1],('B2','ip1'):axes[1,0],('B2','ip5'):axes[1,1]}
            
            self.fig_CO.canvas.resizable       = False
            self.fig_CO.canvas.footer_visible  = False
            self.fig_CO.canvas.header_visible  = False
            self.fig_CO.canvas.toolbar_visible = False
            
            self.CO_lines = {}
            for beam in ['B1','B2']:
                for IP in ['ip1','ip5']:
                    for plane,color in zip(['x','y'],['C0','C1']):
                        self.CO_lines[(beam,IP,plane)], = self.ax_CO[(beam,IP)].plot(twiss[beam][f'mu{plane}']-self.twiss_mu0[(beam,IP,plane)],
                                                                                     np.zeros(len(twiss[beam])),color=color,alpha=0.7,label=f'{plane}-plane')
                    self.ax_CO[(beam,IP)].set_title(f'Beam {beam[-1]}, IP{IP[-1]}',fontsize=8)
                    self.ax_CO[(beam,IP)].set_xlim([-5,5])
                    self.ax_CO[(beam,IP)].set_ylim([-0.15,0.15])
                
                
            self.ax_CO[('B1','ip1')].get_xaxis().set_visible(False)
            self.ax_CO[('B1','ip5')].get_xaxis().set_visible(False)
            
            self.ax_CO[('B1','ip5')].get_yaxis().set_visible(False)
            self.ax_CO[('B2','ip5')].get_yaxis().set_visible(False)
            
            plt.subplots_adjust(hspace = .2,wspace=.001)
            self.ax_CO[('B1','ip1')].legend(loc='lower center',fontsize=8,ncol=2,bbox_to_anchor=(1, 1.15))
            self.fig_CO.supxlabel(r'$\mu_{x,y} - \mu_0$',fontsize=8)
            self.fig_CO.supylabel(r'$\{x,y\}/\sqrt{\epsilon\beta_{x,y}}$',fontsize=8)
            
            display(self.fig_CO.canvas)
        #======================
        
        # Collimator OFFSET
        #==================================
        self.coll_slider = {}
        slider_opt = {'value':0,'min':-self.slider_max,'max':self.slider_max}
        self.coll_slider['t']= widgets.FloatSlider(description=self.slider_unit_txt,**slider_opt,orientation='horizontal',layout=widgets.Layout(display='flex',border=showBorder,width=f'{3/5*self.width}px', height=f'{1/5*self.height}px'),continuous_update=False)
        self.coll_slider['b']= widgets.FloatSlider(description=self.slider_unit_txt,**slider_opt,orientation='horizontal',layout=widgets.Layout(display='flex',border=showBorder,width=f'{3/5*self.width}px', height=f'{1/5*self.height}px'),continuous_update=False)
        self.coll_slider['l']= widgets.FloatSlider(description=self.slider_unit_txt,**slider_opt,orientation='vertical'  ,layout=widgets.Layout(display='flex',border=showBorder,width=f'{1/5*self.width}px', height=f'{3/5*self.height}px'),continuous_update=False)
        self.coll_slider['r']= widgets.FloatSlider(description=self.slider_unit_txt,**slider_opt,orientation='vertical'  ,layout=widgets.Layout(display='flex',border=showBorder,width=f'{1/5*self.width}px', height=f'{3/5*self.height}px'),continuous_update=False)

        for coll in ['l','r','b','t']:
            self.coll_slider[coll].observe(self.update_all_plots,'value')
        #==================================
        
        # GAPS
        #==================================
        self.coll_open = {}
        self.coll_open_units = 1e-3
        self.coll_open['ip1'] = widgets.IntRangeSlider(value=[-15,15],description = 'IP1 [mm]',orientation='horizontal',min=-20,max=20,step=1,continuous_update=False,layout=widgets.Layout(display='flex',border='solid',width=f'{3/5*self.width}px', height=f'{1/5*self.height}px'))
        self.coll_open['ip5'] = widgets.IntRangeSlider(value=[-15,15],description = 'IP5 [mm]',orientation='horizontal',min=-20,max=20,step=1,continuous_update=False,layout=widgets.Layout(display='flex',border='solid',width=f'{3/5*self.width}px', height=f'{1/5*self.height}px'))
        for coll in ['ip1','ip5']:
            self.coll_open[coll].observe(self.update_all_plots,'value')
        
        # Chosen IP
        self.IP = 'ip1'
        self.IP_buttons = {}
        self.IP_buttons['ip1'] = widgets.Button(value='ip1',description='IP1', button_style='info'   , layout=widgets.Layout(width=f'{1/5*self.width}px', height=f'{1/10*self.height}px'),continuous_update=False)
        self.IP_buttons['ip5'] = widgets.Button(value='ip5',description='IP5', button_style='warning',layout=widgets.Layout(width=f'{1/5*self.width}px', height=f'{1/10*self.height}px'),continuous_update=False)

        for IP in ['ip1','ip5']:
            self.IP_buttons[IP].on_click(self.on_IP_click)
        
        #initialize:
        self.coll_ON={}
        self.on_IP_click(self.IP_buttons['ip1'])
        
        self.grid = widgets.GridspecLayout(5, 4, height=f'{1.5*self.height}px',width=f'{2*1.1*self.width}px')
        
        self.grid[1,1] = self.panel_beam
        self.grid[:,3] = self.panel_CO
        self.grid[0,1] = self.coll_slider['t']
        self.grid[1,0] = self.coll_slider['l']
        self.grid[1,2] = self.coll_slider['r']
        self.grid[2,1] = self.coll_slider['b']
        
        self.grid[0,0] = self.IP_buttons['ip1']
        self.grid[0,2] = self.IP_buttons['ip5']
        
        self.grid[3,1] = self.coll_open['ip1']
        self.grid[4,1] = self.coll_open['ip5']
        
        
        # RESET BUTTON
        self.reset_button = widgets.Button(value='reset',description='RESET', button_style='danger',layout=widgets.Layout(width=f'{1/5*self.width}px', height=f'{1/10*self.height}px'),continuous_update=False)
        self.reset_button.on_click(self.on_reset_click)
        self.grid[2,0] = self.reset_button
    
    
    
        
    
    
    @property   
    def wires_pos(self):
        pos_dict = {'l':[self.coll_open['ip5'].value[0]*self.coll_open_units-3e-3,self.slider_unit*self.coll_slider['l'].value],
                    'r':[self.coll_open['ip5'].value[1]*self.coll_open_units+3e-3,self.slider_unit*self.coll_slider['r'].value],
                    't':[self.slider_unit*self.coll_slider['t'].value,self.coll_open['ip1'].value[1]*self.coll_open_units+3e-3],
                    'b':[self.slider_unit*self.coll_slider['b'].value,self.coll_open['ip1'].value[0]*self.coll_open_units-3e-3]}
        return pos_dict
    
    @property   
    def wires_I(self):
        I_dict = {'l':350*self.coll_ON['l'],
                  'r':350*self.coll_ON['r'],
                  't':350*self.coll_ON['t'],
                  'b':350*self.coll_ON['b']}
        return I_dict
    
    def on_reset_click(self,b):
        time.sleep(0.5)
        for coll in ['l','r','b','t']:
            self.coll_slider[coll].value=0
        time.sleep(0.5)
        for coll in ['ip1','ip5']:
            self.coll_open[coll].value  = (-15,15)
        time.sleep(0.5)
            
            
    
    def on_IP_click(self,b):
        self.IP = str(b.description).lower()
        if self.IP == 'ip1':
            for coll,value in zip(['l','r','b','t'],[0,0,1,1]):
                self.coll_ON[coll] = value
        if self.IP == 'ip5':
            for coll,value in zip(['l','r','b','t'],[1,1,0,0]):
                self.coll_ON[coll] = value
        
        self.update_all_plots('')
    
    def update_all_plots(self,_):
        self.update_beam_plot('')
        self.update_co_plot('')
        
    def update_beam_plot(self,_): 
        for coll in ['l','r','b','t']:
            line,value = self.coll_lines[coll],self.coll_slider[coll].value
            if   self.coll_slider[coll].orientation == 'horizontal':
                line.set_xdata(value)
                line.set_alpha(self.coll_ON[coll])
                
            elif self.coll_slider[coll].orientation == 'vertical':
                line.set_ydata(value)
                line.set_alpha(self.coll_ON[coll])
        
        self.fig_beam.canvas.draw()
    
    
    def update_co_plot(self,_):
        
        
        # Beam 1 first
        beam = 'B1'
        kicks = {'x':0,'y':0}
        for loc in ['l','r','b','t']:
            xw,yw = self.wires_pos[loc]
            _Px,_Py = get_kick(self.wires_I[loc],xw,yw,L=1,E=self.E)
            kicks['x'] += _Px
            kicks['y'] += _Py
        

        for plane in ['x','y']:
            co_value = get_CO(mu_s  = twiss[beam][f'mu{plane}'],
                              bet_s = twiss[beam][f'bet{plane}'],
                              P     = kicks[plane],
                              Q     = twiss[beam][f'mu{plane}'][-1],
                              mu0   = self.twiss_mu0[(beam,self.IP,plane)])
            for IP in ['ip1','ip5']:
                self.CO_lines[(beam,IP,plane)].set_ydata(co_value/np.sqrt(self.emitt*twiss[beam][f'bet{plane}']))
        
        # Then Beam 2
        beam = 'B2'
        kicks = {'x':0,'y':0}
        for loc in ['l','r','b','t']:
            xw,yw = self.wires_pos[loc]
            _Px,_Py = get_kick(self.wires_I[loc],-self.B2_dist[(self.IP,'x')] + xw,-self.B2_dist[(self.IP,'y')]+yw,L=1,E=self.E)
            kicks['x'] += _Px
            kicks['y'] += _Py
        

        for plane in ['x','y']:
            
            mu0_from_b1 = np.interp(self.twiss_s[('B1',self.IP)],twiss[beam]['s'],twiss[beam][f'mu{plane}'])
            
            co_value = get_CO(mu_s  = twiss[beam][f'mu{plane}'],
                              bet_s = twiss[beam][f'bet{plane}'],
                              P     = kicks[plane],
                              Q     = twiss[beam][f'mu{plane}'][-1],
                              mu0   = mu0_from_b1)
            
            for IP in ['ip1','ip5']:
                self.CO_lines[(beam,IP,plane)].set_ydata(co_value/np.sqrt(self.emitt*twiss[beam][f'bet{plane}']))
        
        self.fig_CO.canvas.draw()
        #return kicks#co_value
    
    def show(self,):
        display(self.grid)
    
    
# PHYSICS
#=====================================

def get_kick(I,xw,yw,L=1,E=450e9):
    Brho = np.sqrt(E**2-cst.m_p_eV**2)/cst.c

    xObs,yObs = 0,0
    rVec = np.sqrt((xw-xObs)**2 + (yw-yObs)**2)
    amplitude = cst.mu0*(I*L)/(Brho*2*np.pi)/rVec

    Px = amplitude*(xw-xObs)/rVec
    Py = amplitude*(yw-yObs)/rVec

    return Px,Py


def get_CO(mu_s,bet_s,P,Q,mu0):
    # Confirming that mu and bet were given as vectors
    assert(np.shape(mu_s)==np.shape(bet_s))
    
    # Finding beta at location of kick
    bet0 = bet_s[np.argmin(np.abs(mu_s-mu0))] 
    Amp  = np.sqrt(bet_s*bet0/(4*np.sin(np.pi*Q)**2)) * P

    return Amp*np.cos(2*np.pi*(np.abs(mu_s-mu0)) - np.pi*Q)
    
