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

sessions_dir = Path(base_dir, 'research/imaging/datasets/SRPBS/processed_data/pf-pipelines/qunex-nbridge/studies/CANBIND-20220818-mCcU5pi4/sessions')



