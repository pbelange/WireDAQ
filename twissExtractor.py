import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/phbelang/abp/WireDAQ/py_wireDAQ/lhcmask/python_examples/run3_collisions_wire')

import bbcw as bbcw
from cpymad.madx import Madx


def fromOptics(opticsFile = 'opticsfile.1',saveToFolder = None):

    # Importing LHC sequences:
    mad = Madx()
    mad.option(echo = True, warn = True)
    mad.call(file='acc-models-lhc/lhc.seq')
    mad.call(file=f'py_wireDAQ/optics/2021_V6/PROTON/{opticsFile}')

    # Installing wires
    for seq,bv in zip(['lhcb1','lhcb2'],[1,-1]):
        mad.input(f'SEQEDIT, SEQUENCE={seq};FLATTEN;cycle,start=ip3;FLATTEN;ENDEDIT;')
        mad.command.beam(particle='proton',sequence=seq,charge=1,npart=1,energy=450,bv=bv)
        mad.input(f'use,sequence={seq};')
        bbcw.install_wires(mad,{'wires_at_fallback':True,'wires_L':1},seq)


    # Twiss
    twiss = {}
    for seq in ['lhcb1','lhcb2']:
        mad.input(f'use,sequence={seq};')
        mad.twiss()
        twiss[seq[-2:].upper()] = mad.table['twiss'].dframe()
    mad.input('stop;')
    
    if saveToFolder is not None:
        for beam in ['B1','B2']:
            twiss[beam].to_pickle(f"{saveToFolder}/opticsfile{opticsFile.split('.')[-1]}_{beam}_twiss.pkl")

    return twiss


def fromPkl(folder,opticsFile = 'opticsfile.1'):
    twiss = {}
    for beam in ['B1','B2']:
         twiss[beam] = pd.read_pickle(f"{folder}/opticsfile{opticsFile.split('.')[-1]}_{beam}_twiss.pkl")

    return twiss