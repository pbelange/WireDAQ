
import pandas as pd
import numpy as np
from pathlib import Path

from mdutils.mdutils import MdUtils
import WireDAQ.NXCALS as nx

# Creating NXCALS variable containers
#------------------------------------------------
wires     = {'B1': [nx.NXCALSWire(loc = loc) for loc in ['L1B1','L5B1']],
             'B2': [nx.NXCALSWire(loc = loc) for loc in ['R1B2','R5B2']]}
beams     = [nx.NXCALSBeam(name) for name in ['B1','B2']]
LHC       = nx.NXCALSLHC()
b_slots   = np.arange(3564)
#------------------------------------------------

header_file = '/home/phbelang/bblumi/docs/Monitoring/header_index.md'
target_file = '/home/phbelang/bblumi/docs/Monitoring/index.md'

_default_files = '/eos/user/p/phbelang/www/Monitoring_efficiency'
_default_url   = 'https://phbelang.web.cern.ch/Monitoring_efficiency'
_default_bbb_url   = 'https://phbelang.web.cern.ch/Monitoring_bbbsignature'

# Importing fill metadata
#==========================================
df_list = []
for state in ['on', 'off']:
    _df = pd.read_pickle(f'filter_wires_{state}.pkl')
    _df.insert(0, 'Wires', state.upper())
    df_list.append(_df)
df = pd.concat(df_list).sort_index()
#==========================================

# Creating HTML links
#==========================================
DBLM_links = pd.Series(df.index).apply(lambda line:f'DBLM/FILL{line}.html').values
BCTF_links = pd.Series(df.index).apply(lambda line:f'BCTF/FILL{line}.html').values
df.insert(1, 'DBLM', DBLM_links)
df.insert(2, 'BCTF', BCTF_links)
#==========================================


# Creating Markdown file from header file
#==========================================
mdFile = MdUtils(file_name=target_file)
mdFile.read_md_file(file_name=header_file)
mdFile.new_line()


# Creating table
#==========================================
table_header = ["Fill", "Wires status", r"$\beta^*$", "Bunches B1/B2", "Efficiency",'B-by-B signature']
table_content = table_header.copy()
for index, row in df.iterrows():

    new_row  = [f"**{index}**",f"{row['Wires']}",f"{row['HX:BETASTAR_IP1']:.1f} cm" , f"{row[beams[0].Nb]:.0f}/{row[beams[1].Nb]:.0f}"]
    link_str = ''
    if Path(_default_files +'/'+ row['DBLM']).exists():
        link_str += f"[**DBLM**]({_default_url}/{row['DBLM']}){{target=_blank}}"
    else:
        link_str += f"DBLM"
    link_str += ' | '
    if Path(_default_files +'/'+ row['BCTF']).exists():
        link_str += f"[**BCTF**]({_default_url}/{row['BCTF']}){{target=_blank}}"
    else:
        link_str += f"BCTF"
    new_row.append(link_str)
    

    # BbyB signature
    # link_str = ''
    # link_str += f"[**DBLM**]({_default_bbb_url}/{row['DBLM']}){{target=_blank}}"
    # link_str += ' | '
    # link_str += f"BCTF"

    link_str = ''
    if Path(_default_files +'/'+ row['DBLM']).exists():
        link_str += f"[**DBLM**]({_default_bbb_url}/{row['DBLM']}){{target=_blank}}"
    else:
        link_str += f"DBLM"
    link_str += ' | '
    if Path(_default_files +'/'+ row['BCTF']).exists():
        link_str += f"[**BCTF**]({_default_bbb_url}/{row['BCTF']}){{target=_blank}}"
    else:
        link_str += f"BCTF"
    new_row.append(link_str)


    # Sending to table
    table_content.extend(new_row)
mdFile.new_line()

mdFile.new_table(columns=len(table_header), rows=len(df)+1, text=table_content, text_align='center')
#==========================================


# Writing markdown file
#==========================================
mdFile.create_md_file()

#==========================================
print(f'Re-indexed: {target_file}')