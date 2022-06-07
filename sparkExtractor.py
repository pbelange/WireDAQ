import numpy as np
import pandas as pd

import Backend.NXCALS as nx


#=============================================
# Time and variables
#=============================================

# May 22 wire tests
start_time = '2022-05-22 13:00'
end_time   = '2022-05-22 17:30'
savefile   = 'Data/dataset_2022_05_22.parquet'


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
            #df = pd.DataFrame(index=[pd.Timestamp(start_time)],columns = [_var])
            df = pd.DataFrame(columns = [_var])
            return df

    # Converting to UTC time zone:
    df['nxcals_timestamp'] = df['nxcals_timestamp'].apply(lambda t: pd.Timestamp(t).tz_localize('UTC'))    
    df.set_index('nxcals_timestamp', inplace=True)
    df.index.name = 'timestamp'
    df = df.sort_index()
    
    # Extracting value
    df[variable] = df['nxcals_value']
    
    return df[[variable]]
#=============================================



#=============================================
# EXTRACTING AND SAVING
#=============================================
data_list = []
for _var in allvars:
    data_list.append(SparkQuery(_var,start_time,end_time))

data_df = pd.concat(data_list)    
    
# Saving to parquet: 
data_df.to_parquet(savefile, use_deprecated_int96_timestamps=True)
#=============================================
