
import papermill as pm
import pandas as pd


# Choosing output path
output_path = "/home/phbelang/abp/WireDAQ/Results_Lumi_bb/wires_off/"

# Loading desired fills
fills_info       = pd.read_pickle("wires_off.pkl")
fills_to_analyze = list(fills_info.index) #fills_to_analyze = [8033, 8063, 8072, 8076, 8081, 8083, 8088, 8094, 8102, 8103, 8112, 8113, 8115]

for fill in fills_to_analyze:
    params = {"FILL": fill,'wiresON':False,'SAVEPKL' : 'wires_off'}
    pm.execute_notebook(
        "Lumi_bb_analysis.ipynb",
        output_path+f"FILL_{params['FILL']}.ipynb",
        kernel_name="py_wireDAQ",
        parameters=params
    )
