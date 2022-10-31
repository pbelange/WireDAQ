#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 28 11:50:29 2022
i
@author: phbelang
"""

import numpy as np
import pandas as pd

from nxcals import spark_session_builder
from nxcals.api.extraction.data.builders import DevicePropertyDataQuery
from nxcals.api.extraction.data.builders import *


class SparkSession():
    def __init__(self,):
        self.spark = spark_session_builder.get_or_create(app_name='spark-basic', master='yarn')
        
        

    def query(self,variable,start_time,end_time):  
        
        if pd.Timestamp(start_time).tzinfo is not None:
            start_time = str(start_time).split('+')[0]
        if pd.Timestamp(end_time).tzinfo is not None:
            end_time = str(end_time).split('+')[0]
        
        try:
            df = DataQuery.builder(self.spark).byVariables() \
                .system('CMW') \
                .startTime(start_time).endTime(end_time) \
                .variable(variable) \
                .build().toPandas()
        except:
            try:
                df = DataQuery.builder(self.spark).byVariables() \
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