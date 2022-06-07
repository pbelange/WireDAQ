# Wire DAQ

## About
The main tools of this package in `Backend/NXCALS` are a serie of classes used to regroup the NXCALS variable names for BBLR wires and the beam. The idea is to be able to conveniently extract the data from a dataframe following:
```python
import Backend.NXCALSWire as nx
wire = nx.NXCALSWire(loc = 'L1B1')
data = database[[wire['I'],wire['V']]].dropna()

# The above is exactly equivalent to:
data = database[['RPMC.UL14.RBBCW.L1B1:I_MEAS','RPMC.UL14.RBBCW.L1B1:V_MEAS']].dropna()

# One can see all the variable names using:
display(wire)
```

To make it easier to keep track of the units, a dedicated attribute can also be used:
```python
plt.ylabel(f"{wire.label['I']} [{wire.units['I']}]")

# Yields the same as:
plt.ylabel(f"Current [A]")
```



## NXCALS installation:
This package is intended to be used on a acc computer. We first need to install nxcals and the working python environment:
```bash
bash installme.sh 
```

## Data extraction with spark:
It can be convenient to extract the data from spark and save it locally. One can change the `start_time` and `end_time` in `sparkExtractor.py` and then:
```bash
kinit
ipython -i sparkExtractor.py
```

