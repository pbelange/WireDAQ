import numpy as np
import pandas as pd

import Backend.NXCALS_V2 as nx


#=============================================
# Time and variables
#=============================================

# May 22 wire tests
start_time = '2022-08-20 08:27:50'
end_time   = '2022-08-20 20:30:44'
savefile   = 'Dumps_data/F8146.parquet'


# April 8 wire tests
#start_time = '2022-04-08 05:00'
#end_time   = '2022-04-08 16:00'
#savefile   = 'Data/dataset_2022_04_08.parquet'


# Create variable list from 4 wires and 2 beams
#----------------------------------------------
allvars = []
for loc in ['L1B1','R1B2','L5B1','R5B2']:
    wire = nx.NXCALSWire(loc=loc)
    allvars += wire._getVarList()

for name in ['B1','B2']:
    beam = nx.NXCALSBeam(name)
    allvars += beam._getVarList()
    
# for name in ['B1','B2']:
#     LHC = nx.NXCALSLHC()
#     allvars += LHC._getVarList()
#=============================================

    
#=============================================
# SPARK QUERY
#=============================================
from nxcals import spark_session_builder
from nxcals.api.extraction.data.builders import DevicePropertyDataQuery
from nxcals.api.extraction.data.builders import *

spark = spark_session_builder.get_or_create(app_name='spark-basic', master='yarn')

def SparkQuery(variable,start_time,end_time):  
    try:
        df = DataQuery.builder(spark).byVariables() \
            .system('CMW') \
            .startTime(start_time).endTime(end_time) \
            .variable(variable) \
            .build().toPandas()
    except:
        try:
            df = DataQuery.builder(spark).byVariables() \
                .system('WINCCOA') \
                .startTime(start_time).endTime(end_time) \
                .variable(variable) \
                .build().toPandas()
        except:
            # Return empty dataframe is error persists
            return pd.DataFrame(index=[pd.Timestamp(start_time).tz_localize('UTC')],columns = [variable])

    # Converting to UTC time zone:
    df['nxcals_timestamp'] = df['nxcals_timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC'))    
    df.set_index('nxcals_timestamp', inplace=True)
    df.index.name = 'timestamp'
    df = df.sort_index()
    
    # Extracting value
    df[variable] = df['nxcals_value']
    
    if len(df[[variable]]) == 0:
        # For empty dataframe, making sure pd.concat() will go as expected
        return pd.DataFrame(index=[pd.Timestamp(start_time).tz_localize('UTC')],columns = [variable])
    else:     
        return df[[variable]]
#=============================================



#=============================================
# EXTRACTING AND SAVING
#=============================================
data_list = []
# convert to set to eliminate duplicates, then loop
for _var in list(set(allvars)):
    data_list.append(SparkQuery(_var,start_time,end_time))

data_df = pd.concat(data_list,axis=1)    
    
# Saving to parquet: 
data_df.to_parquet(savefile, use_deprecated_int96_timestamps=True)
#=============================================
