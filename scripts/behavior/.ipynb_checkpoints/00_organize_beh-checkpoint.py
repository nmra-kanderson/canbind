#!/bin/python

import yaml
import numpy as np
import pandas as pd
from pathlib import Path



def scale_name_mapping(yaml_file, col_names, scale_metadata):
    """
    TODO
    """
    # read yaml file with variable name mappings
    with open(yaml_file, 'r') as f:
        scale_name_json = yaml.load(f, Loader=yaml.FullLoader)
    item_dict = scale_name_json['items']

    # some variables have aliases that we have to consider
    items_w_alias = scale_metadata.loc[scale_metadata['Aliases'].notna()]

    # dictionary with primary variable as key, comma-sep aliases as values
    alias_dict = {item['ElementName']:item['Aliases'] for i,item in items_w_alias.iterrows()}

    # map the descriptive variables to each of the alises 
    item_dict = add_aliases_to_item_dict(item_dict, alias_dict)

    # map scale items to their descriptive names    
    rename_dict  = match_items_to_newnames(col_names, item_dict)
    return rename_dict


# Set up directories
#base_dir   = '/home/ubuntu/canbind-fsx'
repo_dir   = '/home/ec2-user/SageMaker/suhas/canbind'
clin_dir   = '/home/ec2-user/SageMaker/ebs/fsx/Clinical'
yaml_dir   = Path(repo_dir, 'scripts/reference/behavior/scale_yamls')


scale_list = [
    'ATHF','BMI','BRIAN','CNSVS','DARS','DID','GAD7',
    'IPAQ','MADRS','MINI','PSQI','QIDS','SDS','SHAPS','WHOQOL',
    'BISBAS','BPI','CGI','DEMO','ECRR','HCL32','LEAPS',
    'MEDHIS','NEOFFI','PSYHIS','QLESQ','SEXFX','SPAQ','YMRS'
]

#sessions_dir = Path(base_dir, 'research/imaging/datasets/SRPBS/processed_data/pf-pipelines/qunex-nbridge/studies/CANBIND-20220818-mCcU5pi4/sessions')

# read each behavioral scale
scale_dict = {}
for scale in scale_list: 
    data_list = []
    for grp in ['Control', 'MDD']:
        scale_dir  = Path(clin_dir, scale, grp)
        csv_list   = list(Path(clin_dir, scale, grp).glob('*csv'))
        print(len(csv_list))
        if len(csv_list) == 1:
            scale_df = pd.read_csv(csv_list[0])
            data_list.append(scale_df)
    if len(data_list) > 1:
        scale_df = pd.concat(data_list)
    else: 
        scale_df = data_list[0]
    scale_dict[scale] = scale_df


# rename columns using descriptive ids in reference yamls
rename_scale_dict = {}
for scale in scale_dict.keys():
    print( f'------- {scale} -------')
    scale_df  = scale_dict[scale]
    
    # read yaml
    yaml_file = Path(yaml_dir, f'{scale.lower()}.yaml')    
    with open(yaml_file, 'r') as f:
        scale_name_json = yaml.load(f, Loader=yaml.FullLoader)
    # original id to descipritive id 
    item_dict = scale_name_json['items']
    
    # create a name dictionary for column replacement 
    newname_dict = {}
    for item in scale_df.columns:
        if item in item_dict.keys():
            new_name = f'{scale}-{item}-{item_dict[item]}'
            newname_dict[item] = new_name
        else:
            newname_dict[item] = item
    
    #save the renamed dataframe
    rename_df = scale_df.copy()
    rename_df.columns = rename_df.columns.map(newname_dict)
    rename_scale_dict[scale] = rename_df
        

medhis_df = rename_scale_dict['MEDHIS']

main_df = rename_scale_dict['DEMO']
iter_scales =  [x for x in rename_scale_dict.keys() if x not in ['MEDHIS', 'DEMO']]
for scale in iter_scales:
    print(scale)
    cur_df = rename_scale_dict[scale]
    main_df = main_df.merge(cur_df, on=['SUBJLABEL', 'Group', 'EVENTNAME', 'Visitnum'], how='outer')
    print(main_df.shape)
    
# write combined file
beh_out = '/home/ec2-user/SageMaker/ebs/fsx/organised_raw_data/beh/CANBIND_clinical_baseline.df'
main_df.to_csv(beh_out, index=None)

