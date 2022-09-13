#!/bin/python

import numpy as np
import pandas as pd
from pathlib import Path


# Set up directories
base_dir   = '/home/ubuntu/canbind-fsx'
clin_dir   = '/home/ubuntu/fsx/Clinical'

scale_list = [
    'ATHF','BMI','BRIAN','CNSVS','DARS','DID','GAD7',
    'IPAQ','MADRS','MINI','PSQI','QIDS','SDS','SHAPS','WHOQOL',
    'BISBAS','BPI','CGI','DEMO','ECRR','HCL32','LEAPS',
    'MEDHIS','NEOFFI','PSYHIS','QLESQ','SEXFX','SPAQ','YMRS'
]

for scale in scale_list: 
    print(scale)
    scale_dir  = Path(clin_dir, scale)
    scale_list = []
    for grp in ['Control', 'MDD']:
        csv_list = list(scale_dir.glob(f'{grp}/*csv'))
        if len(csv_list) == 1:
            scale_df = pd.read_csv(csv_list[0])
            scale_list.append(scale_df)
            print(len(csv_list))


sessions_dir = Path(base_dir, 'research/imaging/datasets/SRPBS/processed_data/pf-pipelines/qunex-nbridge/studies/CANBIND-20220818-mCcU5pi4/sessions')



