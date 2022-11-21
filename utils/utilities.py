import os
import re
import pandas as pd
from pathlib import Path

__all__ = [
    'extract_filename_components',
    'compile_production_data',
    'print_feedback',
    'make_directory'
    ]


        
def make_directory(dir_path):
    """
    Make a Directory
    
    Parameters
    ----------
    dir_path: str, Path
        Full path to the directory to make
    
    Returns
    --------
    dir_path: str, Path
        Full path to the directory to make
    """
    if dir_path == str:
        dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path


def extract_filename_components(fname):
    """
    Create a dictionary it key:value descriptors from a
    BIDS formatted filename. 

    Paarameters
    -----------
    fname: str
        BIDS format Filename to parse
            e.g. sub-001_task-rest_study-EMBARC.dtseries.nii

    Returns
    -----------
    bids_dict: dict
        Dictionary where keys are BIDS fields, values are BIDS field entried
            e.g. {'sub': '001', 'study':'EMBARC'}
    """
    # remove file endings
    file_ends = [
        '.csv',
        '.dtseries',
        '.dscalar'
        '.ptseries',
        '.pscalar'
    ]
    fname_base = [fname.split(x)[0] if x in fname else fname for x in file_ends] [0]

    bids_dict = dict()
    splits    = ['sub-', '_task-', '_proc-', '_study-', '_atlas-', '_stat-']
    cur_split = 'sub-'
    for cur_split in splits:
        if cur_split in fname_base:
            post_split = fname_base.split(cur_split)[-1]
            value = re.split('|'.join(splits), post_split)[0]
            value = value.replace('.dtseries.nii','').replace('.ptseries.nii','')
            key   = cur_split.replace('_','').replace('-','')
            bids_dict[key] = value

    combine_keys = [s for s in bids_dict.keys() if s != 'sub']
    uniq_id_list = []
    for key in combine_keys:
        val = bids_dict[key]
        uniq_id_list.append(f'_{key}-{val}')
    uniq_id = ''.join(uniq_id_list)[1:].replace('.csv.gz','').replace('csv','')
    bids_dict['uniq_id'] = uniq_id
    return bids_dict


def compile_production_data(data_dir, feature, glob_string, session_dir):
    """
    Compile all of the "refined" data files into a single
    production-ready format. Data are subject(row)xfeature(col).

    Parameters
    -----------
    data_dir: str
        Full path to the "refined" data directory
    feature: str
        Name of the imaging feature to compile
        Example: "PearsonR"
    glob_string: str
        Unique file identifier of the refined data files to 
        compile
        Example: '_stat-PearsonR_long.csv.gz'
    session_dir: pathlib.PosixPath
        Full path to the Qunex session directory
    """
    # path to the intermediate "refined" data elements
    feature_dir = Path(data_dir, session_dir.stem, 'functional', feature)

    # compile individual feature files 
    read_files = [x for x in feature_dir.rglob(f'*{glob_string}')]
    
    df_list = []
    for read_file in read_files:
        try: 
            fname       = str(read_file).split('/')[-1]
            file_parts  = extract_filename_components(fname)
            meta_data   = pd.DataFrame(file_parts, index=[0])
            meta_data.insert(1, 'feature', feature)

            # read long dataset and transpose to wide
            # reading very wide DFs takes a long time
            dat_df   = pd.read_csv(read_file, compression='gzip')
            # if data is a matrix
            if dat_df.shape[0] != 1:
                df_wide  = pd.DataFrame(dat_df.value).transpose().reset_index(drop=True)
                df_wide.columns = dat_df['roi_1'] + '__to__' + dat_df['roi_2']
            else:
                df_wide = dat_df[[x for x in dat_df.columns if x not in ['src_file', 'session_id']]]
            
            df_wide_concat = pd.concat([meta_data, df_wide], axis=1)
            df_list.append(df_wide_concat)
        except: 
            pass
    if len(df_list) > 0:
        feature_df = pd.concat(df_list)
        return feature_df


def print_feedback(print_string):
    """
    Print fancy feedback
    """
    print('------------------------------')
    print(print_string)
    print('------------------------------')


