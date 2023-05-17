# source /home/lumimod/2022_09_29_NXCALS/venv/bin/
from pathlib import Path
import os
import getpass
import logging
import pandas as pd

# NXCALS imports
#---------
import nx2pd as nx 
from nxcals.spark_session_builder import get_or_create, Flavor
#---------


class SparkSession():
    def __init__(self,logfile='./.spark.log'):
        # Configuration details
        self.TZONE    = 'Europe/Paris'
        self.username = getpass.getuser()
        os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"


        # Logging and execution of kinit
        logging.basicConfig(filename=logfile, encoding='utf-8',level=logging.INFO,format='%(message)s')
        logging.info('\n'+40*'=')
        logging.basicConfig(filename=logfile, encoding='utf-8',level=logging.INFO,force=True,format='%(asctime)s %(message)s')
        logging.info('Executing the kinit')
        os.system(f'kinit -f -r 5d -kt {os.path.expanduser("~")}/{getpass.getuser()}.keytab {getpass.getuser()}');
        logging.basicConfig(format='%(asctime)s %(message)s')
        
        # Creating spark instance (using LOCAL instead of YARN_LARGE)
        logging.info('Creating the spark instance')

        _spark = get_or_create(flavor= Flavor.LOCAL,
                                conf= { 'spark.driver.maxResultSize': '8g', 
                                        'spark.executor.memory'     : '8g',
                                        'spark.driver.memory'       : '16g',
                                        'spark.executor.instances'  : '20',
                                        'spark.executor.cores'      : '2',
                                        })
        self.spark  = nx.SparkIt(_spark)
        logging.info('Spark instance created.')

    # To standardize the timestamp from chosen TZONE
    #------------------------------------------------
    def standard_ts(self,ts):
        if pd.Timestamp(ts).tzinfo is None:
            return pd.Timestamp(ts,tz=self.TZONE)
        else:
            return pd.Timestamp(ts).tz_convert(self.TZONE)
    

    # Extract information about a given fill
    #------------------------------------------------
    def fill_info(self,fill):
        _fill_info = self.spark.get_fill_time(fill)
        # shifting end time
        _fill_info['end'] -= pd.Timedelta('1ns')
        return _fill_info


    # Querrying list of variables between start_time and end_time
    #------------------------------------------------
    def query(self,variables,start_time=None,end_time=None,fill=None):
        if all(_v is None for _v in [start_time,end_time,fill]):
            raise ValueError('At least one of start_time, end_time or fill must be provided.')
        if fill is not None:
            _fill_info = self.spark.get_fill_time(fill)
            start_time = self.standard_ts(_fill_info['start'])
            end_time   = self.standard_ts(_fill_info['end'])-pd.Timedelta('1ns')
        else:
            start_time = self.standard_ts(start_time)
            end_time   = self.standard_ts(end_time)


        querry_df = []
        for _var in self.spark.get_variables(variables):
            # Extracting the data
            _df = self.spark.nxcals_df( data    =[_var['name']], 
                                        t0      = start_time,
                                        t1      = end_time,
                                        pandas_processing   = [ nx.pandas_get, 
                                                                nx.pandas_pivot])

            # Cleaning the data
            for _column, _type in zip(_df.columns,_df.dtypes):
                if _type==object:
                    if (len(_df[_column].dropna())>0) and (type(_df[_column].dropna().iloc[0])==dict):
                        _df[_column] = _df[_column].apply(lambda x:x['elements'] if pd.notnull(x) else x)
            
            querry_df.append(_df.sort_index())

        return pd.concat(querry_df, axis=1).sort_index()


