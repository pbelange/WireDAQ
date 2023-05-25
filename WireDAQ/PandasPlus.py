
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# ADDING FUNCTIONS TO BASIC PANDAS OBJECTS
#=================================================
from pandas.core.base import PandasObject


TZONE = 'Europe/Paris'
# General function
def ensure_tz(ts):
    if pd.Timestamp(ts).tzinfo is None:
        ts = pd.Timestamp(ts).tz_localize(TZONE)
    return ts



# PLOTTING TOOL
#=================================================
def nxPlot(self,xlabel,ylabel, *args, **kwargs):
    subset  = self.dropna(subset=[ylabel])
    
    if xlabel == 'index':    
        plt.plot(subset.index,subset[ylabel], *args, **kwargs)
    else:
        plt.plot(subset[xlabel],subset[ylabel], *args, **kwargs)
    
PandasObject.nxPlot = nxPlot
#=================================================


# Querry data
#=================================================
def at_idx(self,target,_var=None,method='nearest',return_idx=False):
    pdMethod = {'nearest':'nearest','previous':'pad','next':'backfill'}[method]
    
    if isinstance(target,(list,type(np.array([])))):
        _idx = self.index[self.index.get_indexer(target, method=pdMethod)]
    else:
        _idx = self.index[self.index.get_indexer([target], method=pdMethod)][0]

    if isinstance(self,(pd.core.series.Series)):
        if return_idx:
            return _idx,self.loc[_idx]
        else:
            return self.loc[_idx]
    else:
        if return_idx:
            return _idx,self.loc[_idx,_var]
        else:
            return self.loc[_idx,_var]

def at_ts(self,target,_var=None,method='nearest',return_ts=False):
    pdMethod = {'nearest':'nearest','previous':'pad','next':'backfill'}[method]
    
    target = ensure_tz(target)

    self_cp = self.set_index('Timestamp',inplace=False)

    if isinstance(target,(list,type(np.array([])))):
        _ts = self_cp.index[self_cp.index.get_indexer(target, method=pdMethod)]
    else:
        _ts = self_cp.index[self_cp.index.get_indexer([target], method=pdMethod)][0]

    if isinstance(self,(pd.core.series.Series)):
        if return_ts:
            return _ts,self_cp.loc[_ts]
        else:
            return self_cp.loc[_ts]
    else:
        if return_ts:
            return _ts,self_cp.loc[_ts,_var]
        else:
            return self_cp.loc[_ts,_var]

PandasObject.at_idx = at_idx
PandasObject.at_ts  = at_ts
#=================================================


# Manipulate data
#=================================================
def bin(self,_var,window=None,bins=None):
    # GROUPING DATA IN TIME WINDOWS
    sub     = self.dropna(subset=[_var])
    if bins is not None:
        grouped = sub.groupby(pd.cut(sub['Time'],bins=bins))
    if window is not None:
        grouped = sub.groupby(pd.cut(sub['Time'],bins=np.arange(sub.iloc[0]['Time'],sub.iloc[-1]['Time'],window)))

    # AVG in each time window
    values   = np.array(grouped[_var].mean())
    bin_time = np.array(pd.Series(grouped.groups.keys()).apply(lambda line:line.mid))


    # UNPACKING RESULTS
    bin_unix      = (bin_time - sub.iloc[0]['Time'])*1e9 + sub.index[0]
    bin_timestamp = [pd.Timestamp(t).tz_localize('UTC').tz_convert(TZONE) for t in bin_unix]
    

    return pd.DataFrame({'unix':bin_unix,'Timestamp':bin_timestamp,'Time':bin_time,_var:values})
PandasObject.bin       = bin
PandasObject.bin_time  = bin
#=================================================

# Manipulate data
#=================================================
def bin_unix(self,_var,bins=None):
    # GROUPING DATA IN TIME WINDOWS
    sub     = self.dropna(subset=[_var])
    grouped = sub.groupby(pd.cut(sub.index,bins=bins))

    # AVG in each time window
    values   = np.array(grouped[_var].mean())
    unix     = np.array(pd.Series(grouped.groups.keys()).apply(lambda line:line.mid))

    return pd.DataFrame({_var:values},index=unix)
PandasObject.bin_unix  = bin_unix
#=================================================


# Manipulate Bunch-by-bunch
#=================================================

# Call: database[var].to_2D()
def to_2D(self,):
    sub = self.dropna()
    return sub.index,np.stack(sub.values)
PandasObject.to_2D  = to_2D


#=================================================