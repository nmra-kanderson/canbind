#!/bin/python

import os
import csv
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from convert_eprime.convert import text_to_csv

def get_encoding(fpath):
    """
    TODO
    """
    cmd = ['file', '-bi', fpath]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    txt_encode = output.decode('utf-8').replace('\n','').split('charset=')[-1].upper()
    return txt_encode


def load_edat_csv(edat_csv):
    """
    TODO
    """
    text_enc = get_encoding(edat_csv)
    df = pd.read_csv(edat_csv, encoding=text_enc, skiprows=0)
    # if first row is filepath, skip
    if '\\' in df.columns.values[0]:
        skiprows = 1
    else: 
        skiprows = 0
    
    with open(edat_csv, 'r', encoding=text_enc) as csvfile:
        dat = csvfile.readline()
        dat = csvfile.readline()
        dat = csvfile.readline()
        dialect = csv.Sniffer().sniff(dat, delimiters='\t,')
    delim = dialect.delimiter
    
    df = pd.read_csv(edat_csv, encoding=text_enc, skiprows=skiprows, sep=delim)
    return df
        
        
def make_anhed_timings(df):
    """
    TODO
    """
    stimuli = [
        'HitNoPrizeFeedback',
        'HitPrizeFeedback',
        'IncentiveCue',
        'NoIncentiveCue',
        'Miss', 
        'MissIncentive',
        'MissNonIncentive',
        'HitIncentive', 
        'HitNonIncentive'
    ]
    stim_names = {
        'IncentiveCue': 'CueIncentive',
        'NoIncentiveCue': 'CueNoIncentive',
        'HitNoPrizeFeedback': 'FbkHitNoPrize',
        'HitPrizeFeedback': 'FbkHitPrize',
        'Miss': 'FbkMiss',
        'HitIncentive': 'FbkHitIncentive',
        'HitNonIncentive': 'FbkHitNonIncentive',
        'MissNonIncentive': 'FbkMissNonIncentive',
        'MissIncentive': 'FbkMissIncentive',
    }

    ######################
    # MISS trials - ALL
    ######################
    df['Miss.OnsetTime'] = df['TooEarlyFeedback.OnsetTime']
    df.loc[df['TooEarlyFeedback.OnsetTime'] == 0, 'Miss.OnsetTime'] = df.loc[df['TooEarlyFeedback.OnsetTime'] == 0, 'TooLateFeedback.OnsetTime']
    
    # MISS trials - Incentivized
    # --------------------------
    df['MissIncentive.OnsetTime'] = df['Miss.OnsetTime']
    df.loc[df['incentive'] != 1, 'MissIncentive.OnsetTime'] = 0

    # MISS trials - Non-Incentivized
    # --------------------------
    df['MissNonIncentive.OnsetTime'] = df['Miss.OnsetTime']
    df.loc[df['incentive'] == 1, 'MissNonIncentive.OnsetTime'] = 0


    ######################
    # HIT trials - ALL
    ######################
    df['Hit.OnsetTime'] = df['HitPrizeFeedback.OnsetTime']
    df.loc[df['HitPrizeFeedback.OnsetTime'] == 0, 'Hit.OnsetTime'] = df.loc[df['HitPrizeFeedback.OnsetTime'] == 0, 'HitNoPrizeFeedback.OnsetTime']

    # HIT trials - Incentivized
    # --------------------------
    df['HitIncentive.OnsetTime'] = df['Hit.OnsetTime']
    df.loc[df['incentive'] != 1, 'HitIncentive.OnsetTime'] = 0

    # HIT trials - Non-Incentivized
    # --------------------------
    df['HitNonIncentive.OnsetTime'] = df['Hit.OnsetTime']
    df.loc[df['incentive'] == 1, 'HitNonIncentive.OnsetTime'] = 0
    ######################

    df.loc[df['incentive'] == 1, 'IncentiveCue.OnsetTime']
    df.loc[df['incentive'] != 1, 'IncentiveCue.OnsetTime']

    df.loc[df['incentive'] == 1, 'NoIncentiveCue.OnsetTime']
    df.loc[df['incentive'] != 1, 'NoIncentiveCue.OnsetTime']


    # generate 3-column FSL timing data
    task_onset = df['IncentiveCue.OnsetTime'].iloc[0]
    timing_dict = {}
    for stim in stimuli: 
        stim_times = df[f'{stim}.OnsetTime']
        time_array = stim_times[stim_times != 0]
        time_sec   = np.round((time_array-task_onset)/1000, 2)
        stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
        timing_dict[stim_names[stim]] = stim_df

    ######################
    # Reaction Time
    ######################
    # combine RTs from the "Box" and "Fill" parts of the trial
    df['RT'] = df['Box.RTTime']
    df.loc[df['Box.RTTime'] == 0, 'RT'] = df.loc[df['Box.RTTime'] == 0, 'Fill.RTTime']

    # split RTs by incentivized / non-incentivized RTs
    incentive_rt_times    = df.loc[df['incentive'] == 1, 'RT']

    # incentive
    incentive_rt_times    = incentive_rt_times[incentive_rt_times != 0]
    incentive_rt_times_sec = np.round((incentive_rt_times-task_onset)/1000, 2)
    stim_df    = pd.DataFrame({'onset':incentive_rt_times_sec, 'len': 1, 'mag':1})
    timing_dict['BoxRTIncentive'] = stim_df

    # non-incentive 
    nonincentive_rt_times    = df.loc[df['incentive'] == 0, 'RT']
    nonincentive_rt_times  = nonincentive_rt_times[nonincentive_rt_times != 0]
    nonincentive_rt_times = np.round((nonincentive_rt_times-task_onset)/1000, 2)
    stim_df    = pd.DataFrame({'onset':nonincentive_rt_times, 'len': 1, 'mag':1})
    timing_dict['BoxRTNoIncentive'] = stim_df
    return timing_dict
    

def make_gonogo_timings(df):
    """
    TODO
    """
    # get rid of rest/fix trials
    #df = df.loc[df['Procedure'] not in ['rest','instruct']]
    df_filt = df.loc[~df['Procedure'].isin(['rest'])]
    
    # we're assuming the 28second fixation cross is the experiment start
    task_onset  = df['fixate28s.OnsetTime'].iloc[0]
    thankstime  = df['endthanks.OnsetTime'].iloc[0]
    
    # model error trials separately
    error_trials = (df_filt.filter(regex='ACC') == 0).sum(axis=1)
    df_noerr = df_filt[~error_trials.astype(bool)]
    df_err   = df_filt[error_trials.astype(bool)]
    
    block_time_dict = {}
    block_dict = {
        'AngryGo': 'trialsGoA',
        'AngryNoGo': 'trialsNoGoA',
        'NeutralGo': 'trialsGoN',
        'NeutralNoGo': 'trialsNoGoN'
    }
    # NoGo blocks
    for block_type in block_dict.keys():
        print(block_type)
        block_df   = df_noerr.loc[df_noerr['Procedure'] == block_dict[block_type]]
        # correct commission
        commission_df = block_df.loc[block_df[f'{block_type}.CRESP'] == 2]
        stim_times    = commission_df['shape.OnsetTime']
        time_sec      = np.round((stim_times-task_onset)/1000, 2)
        stim_df       = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
        block_time_dict[f'{block_type}_correctPress'] = stim_df

        # correct omission
        if 'NoGo' in block_type:
            ommission_df  = block_df.loc[block_df[f'{block_type}.CRESP'].isna()]
            stim_times    = ommission_df['shape.OnsetTime']
            time_sec      = np.round((stim_times-task_onset)/1000, 2)
            stim_df       = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
            block_time_dict[f'{block_type}_correctOmit'] = stim_df

    # model error trials
    stim_times = df_err['shape.OnsetTime']
    time_sec   = np.round((stim_times-task_onset)/1000, 2)
    stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
    block_time_dict['errors'] = stim_df
    
    # model block onsets
    df_onset   = df_filt.loc[df_filt['Procedure'] == 'instruct']
    stim_times = df_onset['instruction.OnsetTime']
    time_sec   = np.round((stim_times-task_onset)/1000, 2)
    stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
    block_time_dict['blockOnsets'] = stim_df
    
    return block_time_dict
    
    
def make_faces_timing(df):
    """
    TODO
    """
    # sort correct vs incorrect responses
    df['ResponseAccuracy'] = [1 if x else 0 for x in df['Face.RESP'] == df['CorrectResponse']]
    
    # trial onset 
    trial_onset = df['Fixation.OnsetTime'].iloc[0]
    
    # elaborate on trial types
    df['FaceType']  = df['HorF'].map({'F':'Fear', 'H':'Happy'})
    df['TrialType'] = df['CorI'].map({'I':'Incon', 'C':'Con'})
    
    # identify cI, cC, iC, iI trial pairs
    pair_list = [ df['TrialType'].iloc[x-1]+df['TrialType'].iloc[x] for x in np.arange(1,len(df['TrialType']))]
    df['TrialPair'] = ['First'] + pair_list
    
    # mark error trials
    df.loc[df['ResponseAccuracy'] == 0, 'TrialPair'] = 'Error'
    
    timing_dict = {}
    trial_types = ['First', 'ConCon', 'ConIncon', 'InconCon', 'InconIncon', 'Error']
    for trial_type in trial_types:
        print(trial_type)
        stim_times = df.loc[df['TrialPair'] == trial_type, 'Face.OnsetTime']
        time_sec   = np.round((stim_times - trial_onset)/1000, 2)
        stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
        timing_dict[trial_type] = stim_df
    return timing_dict 



#############
#### Set-Up
#############
#data_dir = '/home/ec2-user/SageMaker/ebs/fsx'
data_dir = '/home/ubuntu/canbind_upload'
out_dir  = '/home/ubuntu/canbind-fsx/research/imaging/datasets/CANBIND/organized_raw_data/mri/fmri_task_timings'
#out_dir  = '/home/ec2-user/SageMaker/ebs/fsx/organised_raw_data/mri/fmri_task_timings'

tasks = [
    'fMRI-Anhed',
    'fMRI-Faces',
    'fMRI-GoNoGo',
]

##########
#### MID
##########
edat_csvs = list(Path(data_dir, 'fMRI-Anhed').glob('*/EDATs-*o-*/*csv'))

# the edat csv files have inconsistent encoding
# handle conversions into common utf-8 format
task    = 'MID'
#out_dir = '/home/ec2-user/SageMaker/ebs/fsx/fMRI-Anhed/MDD/fsl_timing_files'
write_dir = Path(out_dir, task)
write_dir.mkdir(exist_ok=True)
os.system(f'sudo chmod 777 {write_dir}')
for edat_csv in edat_csvs:
    # extract session info 
    session_id = ''.join(edat_csv.stem.split('_')[1:3])
    
    # read edat file to dataframe
    df = load_edat_csv(edat_csv)
    print(df.shape)

    # extract timing information 
    time_dict = make_anhed_timings(df)
    
    # write FSL formatted timing info to file
    for key in time_dict.keys():
        time_df    = time_dict[key]
        out_fname  = f'sub-{session_id}_ses-01_task-MID_contrast-{key}.txt'
        write_file = Path(write_dir, out_fname)
        time_df.to_csv(write_file, sep='\t', index=None, header=None)
        
        
        
##############
#### GONOGO
##############
task      = 'GoNoGo'
write_dir = Path(out_dir, task)
write_dir.mkdir(exist_ok=True)
os.system(f'sudo chmod 777 {write_dir}')

edat_csvs = list(Path(data_dir, 'fMRI-GoNoGo').glob('*/EDATs-*o-*/*csv'))

edat_csv = Path('/home/ubuntu/canbind_upload/fMRI-GoNoGo/MDD/EDATs-To-csv/CBN01_MCU_0029_01_SE01_MR_gonogo_trialA.csv')

for edat_csv in edat_csvs:
    # extract session info 
    session_id = ''.join(edat_csv.stem.split('_')[1:3])
    trial      = edat_csv.stem.split('_')[-1]
    
    # read edat file to dataframe
    df = load_edat_csv(edat_csv)
    print(df.shape)
    
    time_dict = make_gonogo_timings(df)    
    
    # write FSL formatted timing info to file
    for key in time_dict.keys():
        time_df    = time_dict[key]
        out_fname  = f'sub-{session_id}_ses-01_task-MID_contrast-{key}.txt'
        write_file = Path(write_dir, out_fname)
        time_df.to_csv(write_file, sep='\t', index=None, header=None)
        
        
        
    
#################
#### Faces Task
#################
edat_csvs = list(Path(data_dir, 'fMRI-Faces').glob('*/EDATs-*o-*/*csv'))
edat_csv  = Path('/home/ubuntu/canbind_upload/fMRI-Faces/Control/EDATs-to-csv/CBN01_CAM_0002_01_SE01_MR_Faces1.csv')

for edat_csv in edat_csvs:
    # extract session info 
    session_id = ''.join(edat_csv.stem.split('_')[1:3])
    trial      = edat_csv.stem.split('_')[-1]
    print(session_id)
    
    # read edat file to dataframe
    df = load_edat_csv(edat_csv)
    if df.shape[0] != 72:
        print(edat_csv)
        print(df.shape)
        print(df['Face.RT'].mean())
    
    

def make_faces_timing(df):
    """
    TODO
    """
    # sort correct vs incorrect responses
    df['ResponseAccuracy'] = [1 if x else 0 for x in df['Face.RESP'] == df['CorrectResponse']]
    
    # trial onset 
    trial_onset = df['Fixation.OnsetTime'].iloc[0]
    
    # elaborate on trial types
    df['FaceType']  = df['HorF'].map({'F':'Fear', 'H':'Happy'})
    df['TrialType'] = df['CorI'].map({'I':'Incon', 'C':'Con'})
    
    # identify cI, cC, iC, iI trial pairs
    pair_list = [ df['TrialType'].iloc[x-1]+df['TrialType'].iloc[x] for x in np.arange(1,len(df['TrialType']))]
    df['TrialPair'] = ['First'] + pair_list
    
    # mark error trials
    #df.loc[df['ResponseAccuracy'] == 0, 'TrialPair'] = 'Error'
    
    timing_dict = {}
    trial_types = ['First', 'ConCon', 'ConIncon', 'InconCon', 'InconIncon', 'Error']
    for trial_type in trial_types:
        print(trial_type)
        stim_times = df.loc[df['TrialPair'] == trial_type, 'Face.OnsetTime']
        time_sec   = np.round((stim_times - trial_onset)/1000, 2)
        stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
        timing_dict[trial_type] = stim_df
    return timing_dict 