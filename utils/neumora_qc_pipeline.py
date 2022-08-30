import os
import subprocess
import numpy as np
import pandas as pd
import nibabel as nb
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from typing import List
import tqdm
import click
from qatoolspython import qatoolspython
from neuroimage import parse_session_hcp, get_qunex_dirs, print_feedback
from IPython.core.debugger import set_trace

@click.command()
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-dir')
@click.option('--scans', multiple=True)
@click.option('--n-processes', type=int, default=4)
@click.option('--regex-session-filter', default=None)
@click.option('--freesurfer-home', default=None)
@click.option('--study-name', type=str)
@click.option('--overwrite', type=bool, default=True,envvar='OVERWRITE')
def main(
        sessions_dir: str,
        output_dir: str,
        regex_session_filter: str,
        n_processes: int,
        study_name: str,
        scans: List[str],
        freesurfer_home: str,
        overwrite: bool
    ):
    print(f'Sessions Dir:          {sessions_dir}')
    print(f'Output Dir:            {output_dir}')
    print(f'Regex Session Filter:  {regex_session_filter}')
    print(f'Scans:                 {scans}')
    print(f'Study Name:            {study_name}')
    print(f'Freesurfer Home:       {freesurfer_home}')
    print(f'N Processes:           {n_processes}')

    tmp = False
    if tmp == True: 
        output_dir = '/fmri-qunex/research/imaging/datasets/embarc/imaging-features/embarc-20220301-EuKEp5Gw'
        sessions_dir = '/fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20220301-EuKEp5Gw/sessions'
        regex_session_filter = '(.{6}_(baseline|wk1))'
        study_name = 'EMBARC'
        n_processes = 24
        scans = ['bold rest run-1', 'bold rest run-2', 'bold reward', 'bold ert']

    # Define output directories (Create if needed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # top-level output directory for raw/refined/production
    study_output_dir = Path(output_dir, study_name)
    study_output_dir.mkdir(parents=True, exist_ok=True)

    # production specific data output directory
    prod_output_dir = Path(study_output_dir, 'production', 'qc')
    prod_output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of QuNex subdirectories for processing
    session_dirs = get_qunex_dirs(sessions_dir, regex_session_filter)
    
    os.environ['FREESURFER_HOME'] = freesurfer_home

    #set_trace()

    # --------------------
    # Compile information about BOLD masks
    # --------------------
    print_feedback('Compiling BOLD Mask Statistics')
    with Pool(n_processes) as pool:
        parallelized_function = partial(
                compile_mask_stats, scans, study_output_dir, study_name)

        # Process results in parallel, and show a progress bar via tqdm
        mask_stats = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                        total=len(session_dirs)))

    # concat into single dataframe
    mask_stats_df = pd.concat(mask_stats)
    # save compiled DF to production directory
    mask_file = Path(prod_output_dir, f'study-{study_name}_bold_mask.csv.gz')
    mask_stats_df.to_csv(mask_file, index=None, compression='gzip')

    #set_trace()
    
    # --------------------
    # Compile BOLD head motion estimates 
    # --------------------
    print_feedback('Compiling BOLD Motion Statistics')
    with Pool(n_processes) as pool:
        parallelized_function = partial(
                read_bold_motion, scans, study_output_dir, study_name)
        # Process results in parallel, and show a progress bar via tqdm
        motion_stats = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                        total=len(session_dirs)))

    # concatenate into single DF 
    motion_stats_df = pd.concat(motion_stats)
    # save compiled head motion DF to production dir/disk
    motion_file = Path(prod_output_dir, f'study-{study_name}_bold_motion.csv.gz')
    motion_stats_df.to_csv(motion_file, index=None, compression='gzip')


    # --------------------
    # Freesurfer QA tools
    # --------------------
    #set_trace()
    print_feedback('Freesurfer QA tools')
    for session_dir in session_dirs:
        try:
            freesurfer_qa(study_output_dir, overwrite, session_dir)
        except Exception as e:
            print(e)

    qa_list = []
    for session_dir in session_dirs:
        raw_dir = Path(study_output_dir, 'raw', session_dir.stem, 'qc')
        if Path(raw_dir, 'qatools-results.csv').exists():
            qa_df   = pd.read_csv(Path(raw_dir, 'qatools-results.csv'))
            qa_list.append(qa_df)

    # concatenate into single DF 
    fs_qa_df = pd.concat(qa_list)
    fs_qa_df.insert(0, 'session_id', fs_qa_df['subject'])
    #fs_qa_df.insert(0, 'participant_id', [x.split('_')[0] for x in fs_qa_df['session_id']])
    #fs_qa_df.insert(0, 'visit', [x.split('_')[1] for x in fs_qa_df['session_id']])
    # save compiled head motion DF to production dir/disk
    qa_file = Path(prod_output_dir, f'study-{study_name}_freesurfer_qa.csv.gz')
    fs_qa_df.to_csv(qa_file, index=None, compression='gzip')


def freesurfer_qa(study_output_dir, overwrite, session_dir): 
    """
    TODO 
    """
    session_id = session_dir.stem
    
    # RAW data output dir
    raw_dir = Path(study_output_dir, 'raw', session_id, 'qc')
    raw_dir.mkdir(exist_ok=True, parents=True)
    out_file = Path(raw_dir, 'qatools-results.csv')
    
    try: 
        if overwrite == True or out_file.exists() == False:
            subjects_dir = Path(session_dir, f'hcp/{session_id}/T1w')
            if Path(subjects_dir, session_id, 'stats/aseg.stats').exists():
                qatoolspython.run_qatools(subjects_dir=str(subjects_dir),
                    output_dir=str(raw_dir),
                    subjects=[session_id])
            #qatool_cmd = [
            #    'qatools.py',
            #    '--subjects_dir', str(subjects_dir),
            #    '--output_dir', str(raw_dir),
            #    '--subjects', str(session_id)
            #]
            #os.system(' '.join(qatool_cmd) + ' >/dev/null 2>&1')
            #subprocess.call(qatool_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            #qa_df = pd.read_csv(Path(raw_dir, 'qatools-results.csv'))
            return None
        else:
            return None
    except Exception as e:
        print(f'Failed: {session_id}, e - {e}')
        return None

        
def read_bold_motion(scans, study_output_dir, study_name, session_dir):
    '''
    Read motion estimates from QUNEX formatted bold runs.
    Includes measures of relative and absolute head motion and DVARS.

    Parameters
    ----------
    scans: list[str]
        The names of scans to process. The names must exactly match
        what appears in the session_hcp.txt file. Not all scan types
        need to appear in all of the session_hcp.txt files from every
        session, but "ert" won't pick up "bold ert", if "bold ert"
        is what appears in session_hcp.txt
        Example: '['bold ert', 'bold rest run-1', 'bold rest run-2',
            'bold reward']
    study_output_folder: str
        Where to save the outputs of this script
        Example: /fmri-qunex/research/imaging/datasets/embarc/qunex_features/EMBARC
    study_name: str
        A descriptive name of the imaging project/QuNex output being processed
        Example: EMBARC
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions

    Returns
    ----------
    pd.DataFrame
        Dataframe with mask statistics for each scans in this session
    """
    '''
    try: 
        def _read_motion(read_path):
            read_path = Path(read_path)
            if read_path.exists():
                with open(read_path, 'r') as f:
                    dat = f.readlines()
                return_dat = float(dat[0].replace('\n',''))
            else:
                return_dat = np.nan
            return return_dat

        session_id = session_dir.stem

        # RAW data output dir
        session_output_dir = Path(study_output_dir, 'raw', session_id, 'qc')
        session_output_dir.mkdir(exist_ok=True, parents=True)

        refined_output_dir = Path(study_output_dir, 'refined', session_id, 'qc')
        refined_output_dir.mkdir(exist_ok=True, parents=True)

        # bold number to name dictionaries
        scan_dict     = parse_session_hcp(Path(session_dir, 'session_hcp.txt'), scan_types=scans)
        scan_dict_rev = {scan_dict[key]:key for key in scan_dict.keys()}

        # directory with movement data
        mvmt_dir = Path(session_dir, 'images/functional/movement') 

        # read the relative/absolute motion for each run
        est_list = []
        for key in scan_dict.keys():
            bold_num = scan_dict[key]
                    
            motion_dict = {
                'id': session_dir.stem,
                'scan': bold_num,
                'scan_name': scan_dict_rev[bold_num]
            }
            
            # read TSNR
            tsnr_file = Path(session_dir, 'hcp', session_id, 'MNINonLinear/Results', bold_num, f'{bold_num}_Atlas_TSNR.dscalar.nii')
            if tsnr_file.exists():
                tsnr = nb.load(tsnr_file).dataobj[0]
                mean_tsnr = np.nanmean(tsnr)
                motion_dict['tsnr'] = mean_tsnr

            # MNINonLinear results directory for this BOLD run
            result_dir = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/Results', str(bold_num))
            motion_txt_files = ['Movement_AbsoluteRMS_mean.txt', 'Movement_RelativeRMS_mean.txt']        
            for motion_file in motion_txt_files:
                est = _read_motion(Path(result_dir, motion_file))
                motion_type = motion_file.replace('.txt','')
                motion_dict[motion_type] = est

            bold_mvt = Path(mvmt_dir, f'bold{bold_num}.bstats')
            if bold_mvt.exists():
                motion_stats_df = pd.read_table(bold_mvt, delim_whitespace=True)
                dvars           = motion_stats_df.loc[motion_stats_df['frame'] == '#mean', 'dvars'].tolist()[0]
                motion_dict['dvars'] = dvars

            bold_scrub = Path(mvmt_dir, f'bold{bold_num}.scrub')
            if bold_scrub.exists():
                bold_scrub_df   = pd.read_table(bold_scrub, delim_whitespace=True, comment='#')
                censored_frames = np.sum(bold_scrub_df['use'] == 0)
                total_frames    = bold_scrub_df.shape[0]
                motion_dict['percent_censored'] = censored_frames/total_frames

            motion_series   = pd.Series(motion_dict)
            est_list.append(motion_series)

        if len(est_list) != 0:
            motion_df = pd.DataFrame(est_list)

            # save to disk
            motion_file = Path(refined_output_dir, f'sub-{session_id}_study-{study_name}_bold_motion.csv.gz')
            motion_df.to_csv(motion_file, index=None, compression='gzip')

            return motion_df
    except: 
        print(f'Failed: {session_dir}')



def compile_mask_stats(scans, study_output_dir, study_name, session_dir):
    """
    Compile stats about the anatomical mask applied to a BOLD EPI scan.
    These values can be used for later quantitative QC to identify
    misalignment of functional and anatomical data. 

    Parameters
    ----------
    scans: list[str]
        The names of scans to process. The names must exactly match
        what appears in the session_hcp.txt file. Not all scan types
        need to appear in all of the session_hcp.txt files from every
        session, but "ert" won't pick up "bold ert", if "bold ert"
        is what appears in session_hcp.txt
        Example: '['bold ert', 'bold rest run-1', 'bold rest run-2',
            'bold reward']
    study_output_folder: str
        Where to save the outputs of this script
        Example: /fmri-qunex/research/imaging/datasets/embarc/qunex_features/EMBARC
    study_name: str
        A descriptive name of the imaging project/QuNex output being processed
        Example: EMBARC
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions

    Returns
    ----------
    pd.DataFrame
        Dataframe with mask statistics for each scans in this session
    """
    try: 
        # bold number to name dictionaries
        scan_dict = parse_session_hcp(Path(session_dir, 'session_hcp.txt'), scan_types=scans)

        # create subjects specific output directory in the /raw subfolder
        session_id = session_dir.stem
        session_output_dir = Path(study_output_dir, 'raw', session_id, 'qc')
        session_output_dir.mkdir(exist_ok=True, parents=True)

        refined_output_dir = Path(study_output_dir, 'refined', session_id, 'qc')
        refined_output_dir.mkdir(exist_ok=True, parents=True)


        # read stats about EPI mask 
        
        mask_df = read_mask_stats(session_dir, scan_dict)

        # save individual data to file
        for i,mask_row in mask_df.iterrows():
            task      = mask_row['scan_name'].replace('bold', '').replace(' ','').replace('-','_')
            mask_file = Path(session_output_dir, f'sub-{session_id}_task-{task}_study-{study_name}_bold_mask.csv.gz')
            mask_row.to_csv(mask_file, index=None, compression='gzip')
        
        # save refined to disk
        mask_file = Path(refined_output_dir, f'sub-{session_id}_study-{study_name}_bold_mask.csv.gz')
        mask_df.to_csv(mask_file, index=None, compression='gzip')

        return mask_df
    except: 
        print(f'Failed: {session_dir}')


def read_mask_stats(session_dir, bold_dict):
    '''
    Read statistics about mask coverage for EPI BOLD data.
    Assumes QuNex formatted sessions directory.

    Parameters
    ----------
    session_dir: pathlib.PosixPath
        Path to the individual QuNex output directory
    bold_dict: dict
        Bold information produced by "parse_session_hcp"
    
    Returns
    ----------
    mask_df: pd.DataFrame
    '''

    # map BOLD run numbers to scan names
    rev_bold_dict = dict(zip(bold_dict.values(), bold_dict.keys()))

    mask_list = []
    for key in bold_dict.keys():
        bold_num = bold_dict[key]
        
        # MNINonLinear results directory for this BOLD run
        result_dir = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/Results', str(bold_num))
        mask_path  = Path(result_dir, '{}_finalmask.stats.txt'.format(bold_num))
        
        if mask_path.exists():
            stats_df = pd.read_csv(mask_path)
            stats_df.insert(0, 'bold', bold_num)
            stats_df.insert(0, 'scan_name', rev_bold_dict[bold_num])
            stats_df.insert(0, 'id', session_dir.stem)
            mask_list.append(stats_df)
    if len(mask_list) > 0:
        mask_df = pd.concat(mask_list)
        return mask_df


if __name__ == '__main__':
    main()
