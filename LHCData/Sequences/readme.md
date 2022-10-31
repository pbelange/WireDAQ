
Downloaded sequences from  https://gitlab.cern.ch/acc-models/acc-models-lhc/-/tree/2022/

Date: Sept. 27, 2022, permalinks:
https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/96bfc34aee151b08a648539c7fe4aeccce6bb07e/lhc.seq
https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/96bfc34aee151b08a648539c7fe4aeccce6bb07e/lhcb4.seq

# Check if RFCAVITY PROPERLY INSTALLED:
```bash
sed -n 771p lhc.seq
sed -n 773p lhcb4.seq
```
If not, you can add the harmonic numbers with:
```bash
sed -i "s|ACSCA : RFCAVITY, L := l.ACSCA;|ACSCA : RFCAVITY, L := l.ACSCA, HARMON := HRF400;|" acc-models-lhc/lhc.seq
sed -i "s|ACSCA : RFCAVITY, L := l.ACSCA;|ACSCA : RFCAVITY, L := l.ACSCA, HARMON := HRF400;|" acc-models-lhc/lhcb4.seq
```