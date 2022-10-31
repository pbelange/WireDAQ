
import papermill as pm
import pandas as pd


# Choosing output path
output_path = "/home/phbelang/abp/WireDAQ/Results_Lumi_V2/wires_on/"

# Loading desired fills
fills_info       = pd.read_pickle("wires_on.pkl")
fills_to_analyze = list(fills_info.index) #fills_to_analyze = [8033, 8063, 8072, 8076, 8081, 8083, 8088, 8094, 8102, 8103, 8112, 8113, 8115]


#################
#FILL     = 8142
#wiresON  = True
#SAVING   = True
#SAVEPATH = '../Results_Lumi_V2/wires_on/'
########################################


for fill in fills_to_analyze:
    params = {"FILL": fill,'wiresON':True,'SAVING':True,'SAVEPATH' : '../Results_Lumi_V2/wires_on/'}
    pm.execute_notebook(
        "complete_analysis.ipynb",
        output_path+f"FILL_{params['FILL']}.ipynb",
        kernel_name="py_wireDAQ",
        parameters=params
    )
