import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('py_wireDAQ/lhcmask/python_examples/run3_collisions_wire')

import bbcw as bbcw
from cpymad.madx import Madx


def fromOptics(opticsFile = 'opticsfile.1'):

    # Importing LHC sequences:
    mad = Madx()
    mad.option(echo = True, warn = True)
    mad.call(file='/afs/cern.ch/eng/acc-models/lhc/current/lhc.seq')
    mad.call(file=f'/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/{opticsFile}')

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

    return twiss
    
