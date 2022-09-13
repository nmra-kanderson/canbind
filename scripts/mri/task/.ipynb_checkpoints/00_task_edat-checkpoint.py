#!/bin/python

import subprocess
from convert_eprime.convert import text_to_csv

def get_encoding(fpath):
    cmd = ['file', '-bi', fpath]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    txt_encode = output.decode('utf-8').replace('\n','').split('charset=')[-1].upper()
    return txt_encode

    txt_encode = output.decode('utf-8').replace('\n','').split('charset=')[-1].upper()


def load_edat_csv(edat_csv):
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
    
    stimuli = [
        'Box',
        'HitNoPrizeFeedback',
        'HitPrizeFeedback',
        'IncentiveCue',
        'NoIncentiveCue',
        'TooEarlyFeedback',
        'TooLateFeedback'
    ]
    stim_names = {
        'IncentiveCue': 'CueIncentive',
        'NoIncentiveCue': 'CueNoIncentive',
        'Box': 'Target', 
        'HitNoPrizeFeedback': 'FbkHitNoPrize',
        'HitPrizeFeedback': 'FbkHitPrize',
        'TooEarlyFeedback': 'FbkTooEarly',
        'TooLateFeedback': 'FbkTooLate'
    }
    task_onset = df['IncentiveCue.OnsetTime'].iloc[0]
    timing_dict = {}
    for stim in stimuli: 
        stim_times = df[f'{stim}.OnsetTime']
        time_array = stim_times[stim_times != 0]
        time_sec   = np.round((time_array-task_onset)/1000, 2)
        stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
        timing_dict[stim_names[stim]] = stim_df
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
    error_trials = (df.filter(regex='ACC') == 0).sum(axis=1)
    df_noerr = df[~error_trials.astype(bool)]
    df_err   = df[error_trials.astype(bool)]
      
    # create FSL 3 column timing files for each condition 
    block_time_dict = {}
    blocks = ['trialsGoN', 'trialsNoGoN', 'trialsGoA', 'trialsNoGoA']
    for block in blocks: 
        block_df   = df_filt.loc[df['Procedure'] == block]
        stim_times = block_df['shape.OnsetTime']
        time_sec   = np.round((stim_times-task_onset)/1000, 2)
        stim_df    = pd.DataFrame({'onset':time_sec, 'len': 1, 'mag':1})
        block_time_dict[block] = stim_df
    
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
data_dir = '/home/ec2-user/SageMaker/ebs/fsx'
out_dir  = '/home/ec2-user/SageMaker/ebs/fsx/organised_raw_data/mri/fmri_task_timings'

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
out_dir = '/home/ec2-user/SageMaker/ebs/fsx/fMRI-Anhed/MDD/fsl_timing_files'
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
        write_file = Path(out_dir, out_fname)
        time_df.to_csv(write_file, sep='\t', index=None, header=None)
        
        
        
##############
#### GONOGO
##############
edat_csvs = list(Path(data_dir, 'fMRI-GoNoGo').glob('*/EDATs-*o-*/*csv'))
edat_csv = edat_csvs[20]
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
        write_file = Path(out_dir, out_fname)
        time_df.to_csv(write_file, sep='\t', index=None, header=None)
        
        
        
    
#################
#### Faces Task
#################
edat_csvs = list(Path(data_dir, 'fMRI-Faces').glob('*/EDATs-*o-*/*csv'))

edat_csv = edat_csvs[12]

for edat_csv in edat_csvs:
    # extract session info 
    session_id = ''.join(edat_csv.stem.split('_')[1:3])
    trial = edat_csv.stem.split('_')[-1]
    print(session_id)
    
    # read edat file to dataframe
    df = load_edat_csv(edat_csv)
    if df.shape[0] != 72:
        print(edat_csv)
        print(df.shape)
    
    

    