# source /home/lumimod/2022_09_29_NXCALS/venv/bin/
from pathlib import Path
import os
import getpass
import logging
import sys
import nx2pd as nx 
import pandas as pd
import time
import shutil

import yaml

with open('config.yaml','r') as fid:
    configuration = yaml.safe_load(fid)

TZONE = configuration['time_zone']

# Where to store the parquet files
my_path=Path(configuration['parquet_path'])

#from cron_config import datavars

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream='test.log', level=logging.INFO)
logging.basicConfig(filename=configuration['logfile_path'], encoding='utf-8',
                    level=logging.INFO,
                    format='%(message)s')
logging.info('\n'+40*'=')
logging.basicConfig(filename=configuration['logfile_path'], encoding='utf-8',                    level=logging.INFO,force=True,
                    format='%(asctime)s %(message)s')

os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
username = getpass.getuser()
print(f'Assuming that your kerberos keytab is in the home folder, ' 
      f'its name is "{getpass.getuser()}.keytab" '
      f'and that your kerberos login is "{username}".')

logging.info('Executing the kinit')
os.system(f'kinit -f -r 5d -kt {os.path.expanduser("~")}/{getpass.getuser()}.keytab {getpass.getuser()}');

# %%

logging.basicConfig(format='%(asctime)s %(message)s')

from nxcals.spark_session_builder import get_or_create, Flavor
logging.info('Creating the spark instance')
# Using LOCAL instead of YARN_LARGE
spark = get_or_create(flavor=Flavor.LOCAL,
conf={'spark.driver.maxResultSize': '8g', 
    'spark.executor.memory':'8g',
    'spark.driver.memory': '16g',
    'spark.executor.instances': '20',
    'spark.executor.cores': '2',
    })
sk  = nx.SparkIt(spark)
logging.info('Spark instance created.')


data = ['LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:target',
 'ALICE:LUMI_TOT_INST',
 'ATLAS:BUNCH_LUMI_INST',
 'ATLAS:LUMI_TOT_INST',
 'CMS:BUNCH_LUMI_INST',
 'CMS:LUMI_TOT_INST',
 'HX:BETASTAR_IP1',
 'HX:BETASTAR_IP2',
 'HX:BETASTAR_IP5',
 'HX:BETASTAR_IP8',
 'HX:BMODE',
 'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',
 'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',
 'LHC.BCTDC.B6R4.B1:BEAM_INTENSITY',
 'LHC.BCTDC.B6R4.B2:BEAM_INTENSITY',
 'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY',
 'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY',
 'LHC.BCTFR.B6R4.B1:BUNCH_INTENSITY',
 'LHC.BCTFR.B6R4.B2:BUNCH_INTENSITY',
 'LHC.BQM.B1:NO_BUNCHES',
 'LHC.BQM.B2:NO_BUNCHES',
 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H',
 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V',
 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H',
 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V',
 'LHC.STATS:LHC:INJECTION_SCHEME',
 'LHCB:LUMI_TOT_INST',
 'LhcStateTracker:LHCBEAM1:QH_TRIM:value',
 'LhcStateTracker:LHCBEAM1:QV_TRIM:value',
 'LhcStateTracker:LHCBEAM2:QH_TRIM:value',
 'LhcStateTracker:LHCBEAM2:QV_TRIM:value',
 'RPMC.UL14.RBBCW.L1B1:I_MEAS',
 'RPMC.UL16.RBBCW.R1B2:I_MEAS',
 'RPMC.UL557.RBBCW.R5B2:I_MEAS',
 'RPMC.USC55.RBBCW.L5B1:I_MEAS']

def get(t0, t1, data):
    my_var = sk.get_variables(data)
    my_df = []
    for ii in my_var:
        df = sk.nxcals_df([ii['name']], 
                    t0,
                    t1,
                    pandas_processing=[
                        nx.pandas_get, 
                        nx.pandas_pivot])
        for my_column, my_type in zip(df.columns,df.dtypes):
                if my_type==object:
                    if (len(df[my_column].dropna())>0) and (type(df[my_column].dropna().iloc[0])==dict):
                        df[my_column] = df[my_column].apply(lambda x:x['elements'] if pd.notnull(x) else x)
        my_df.append(df.sort_index())
    return pd.concat(my_df, axis=1).sort_index()


t1 = pd.Timestamp.now(tz=TZONE)
t0 = t1 - pd.Timedelta(hours=2)

# Test mode:
if False:
    t1 = pd.Timestamp.now(tz=TZONE) - pd.Timedelta(hours=10) 
    t0 = t1 - pd.Timedelta(hours=2)

my_df = get( t0, t1, data)
my_df.to_parquet(my_path / f'{t1}.parquet')


logging.info(f'Data Saved,[{str(t0)[11:19]} - {str(t1)[11:19]}]'+'\n'+40*'=')

# %%

import glob
import os
if True:
    files = sorted(glob.glob(str(my_path / '*.parquet')))
    if len(files) > 5:
        for file in files[:-5]:
            os.remove(file)
# %%
