import os
import glob
import re
import h5py
import inspect
import pandas as pd
import nibabel as nb
import shutil
import pathlib
import quilt3 
import subprocess
import numpy as np
from scipy.fftpack import fft
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List
import tqdm
import click
import itertools
#from nipype.interfaces.utility import Function
#import nipype.pipeline.engine as pe
import pdb
import IPython
from IPython.core.debugger import set_trace
from datetime import datetime
import logging
from constants import QUILT_REF_PACKAGE,TRESTLE_REGISTRY,MRI_CONVERT,MRI_ANATOMICAL_STATS,FREESURFER_HOME,WB_COMMAND,DRSFC,GRAPH_METRICS_SCRIPT
import warnings
warnings.filterwarnings("ignore")

import utilities
# Download reference data
utilities.download_reference()
import neuroimage
from neuroimage import *
from utilities import *


@click.command()
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-dir')
@click.option('--regex-session-filter', default=None)
@click.option('--scans', multiple=True)
@click.option('--input-scan-pattern', type=str)
@click.option('--concatenate-scans', is_flag=True)
@click.option('--concatenate-scan-name', type=str)
@click.option('--study-name', type=str)
@click.option('--cifti-atlas', type=str)
@click.option('--cifti-atlas-name', type=str)
@click.option('--task-feat', multiple=str, default=[None])
@click.option('--make-refined', type=bool, default=True)
@click.option('--make-production', type=bool, default=False)
@click.option('--n-processes', type=int, default=4)
def main(
        sessions_dir: pathlib.PosixPath,
        output_dir: pathlib.PosixPath,
        regex_session_filter: str,
        scans: List[str],
        input_scan_pattern: str,
        concatenate_scans: bool,
        concatenate_scan_name: str,
        study_name: str,
        cifti_atlas: str,
        cifti_atlas_name: pathlib.PosixPath,
        task_feat: List[str],
        make_refined: bool,
        make_production: bool,
        n_processes: int
        ):
    """
    Produce BOLD functional run features from a QuNex output directory.
    The user has to specify the names of the scans to process, and can 
    optionally concatenate 1 or more runs (e.g. rest-run-1 + rest-run-2). 
    The user must also specify a (preferably whole-brain) CIFTI dlabel 
    to use for imaging feature computation and parcellation. 

    Specifically this script will calculate:
        1. Resting-State Functional Connectivity
        2. Global Brain Connectivity
        3. Amplitude Measures (fALFF, ALFF, RSFA)
        4. Resting-State Functional Covariance
        5. Graph Theory Metrics
        6. [TODO] Regional Homogeneity

    Parameters
    ----------
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    output_folder: str
        Where to save the outputs of this script
        Example: /fmri-qunex/research/imaging/datasets/embarc/qunex_features
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
        Example: '(session_a|session_b|session_c)' to capture only
            'session_a', 'session_b', and 'session_c' sessions
    scans: list[str]
        The names of scans to process. The names must exactly match
        what appears in the session_hcp.txt file. Not all scan types
        need to appear in all of the session_hcp.txt files from every
        session, but "ert" won't pick up "bold ert", if "bold ert"
        is what appears in session_hcp.txt
        Example: '['bold ert', 'bold rest run-1', 'bold rest run-2',
            'bold reward']
    input_scan_pattern: str
        String to identify the QuNex output scan for processing. 
        It is recommended to included the ".dtseries.nii" to avoid
        potentially grabbing multiple filetypes. 
        Example: '_Atlas_s_hpss_res-mVWMWB_lpss.dtseries.nii'
    concatenate_scans: bool
        Should the scans be concatenated (in time)
        Example: True
    concatenate_scan_name: str
        Name of the newly produced concatenated scan name
    study_name: str
        A descriptive name of the imaging project/QuNex output being processed
        Example: EMBARC
    cifti_atlas: str
        [Optional] Path to the CIFTI dlabel.nii file in fslr32k HCP space
        Example: ~/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
    cifti_atlas_name: str
        [Optional] Descriptive name for the CIFTI dlabel.nii file
        Example: CABNP
    task_feat: str
        Name of the feat directory with task info to process.
    n_processes: int
        Number of CPUs to use for parallel computation
        Example: 2
    """
    print(f'Sessions Dir:          {sessions_dir}')
    print(f'Output Dir:            {output_dir}')
    print(f'Regex Session Filter:  {regex_session_filter}')
    print(f'Scans:                 {scans}')
    print(f'Input Scan Pattern     {input_scan_pattern}')
    print(f'Concatenate Scans:     {concatenate_scans}')
    print(f'Concatenate Scan Name: {concatenate_scan_name}')
    print(f'Study Name:            {study_name}')
    print(f'Cifti Atlas:           {cifti_atlas}')
    print(f'Cifti Atlas Name:      {cifti_atlas_name}')
    print(f'Task Feat:             {task_feat}')
    print(f'N Processes:           {n_processes}')

    set_trace()

    
    # Create output folder if it doesn't exist
    output_dir       = Path(output_dir)
    study_output_dir = Path(output_dir, study_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of QuNex subdirectories for processing
    session_dirs = get_qunex_dirs(sessions_dir, regex_session_filter)

    # Stage 1: Make Raw and Refined stats files and features
    # --------------------
    #set_trace()
    #IPython.embed()
    if make_refined: 
        print_feedback('Producing "raw" and "refined" stats files')
        with Pool(n_processes) as pool:
            parallelized_function = partial(
                process_session, 
                    study_output_dir, scans, input_scan_pattern, concatenate_scans,
                    concatenate_scan_name, study_name, cifti_atlas, cifti_atlas_name, task_feat)

            # Process results in parallel, and show a progress bar via tqdm
            _ = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                            total=len(session_dirs)))

    # Stage 2: Concatenate all the "refined" stats files into single big csv
    # -------------------
    #set_trace()
    #IPython.embed()
    if make_production: 
        print_feedback('Compiling "production" stats files')

        grep_tuples = (
            ('funccon/covariance', f'_atlas-{cifti_atlas_name}_stat-covariance.pconn.nii'),
            ('funccon/pearsonR', f'_atlas-{cifti_atlas_name}_stat-pearsonR.pconn.nii'),
            ('funccon/pearsonRtoZ', f'_atlas-{cifti_atlas_name}_stat-pearsonRtoZ.pconn.nii'),
        )
        for feature, glob_string in grep_tuples:
            print(f'Working on: {feature}')
            #glob_string = grep_dict[feature]
            feature_name = feature.split('/')[-1]
            feature_folder = feature.split('/')[0]
            refined_dir = Path(output_dir, study_name, 'refined')
            # path to the output directory
            production_dir = Path(output_dir, study_name, 'production', 'functional', feature_folder, feature_name)
            production_dir.mkdir(exist_ok=True, parents=True)


            #x,y,z = compile_production_matrix_data(refined_dir, feature, glob_string, session_dirs[0])
            with Pool(n_processes) as pool:
                parallelized_function = partial(
                    stack_pconn_matrices, 
                    refined_dir, feature, glob_string
                    )
                feature_df_list = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                                total=len(session_dirs)))
                
            # format for hd5 file export
            matrix_list = [x[0] for x in feature_df_list if x is not None]
            mat_stack   = np.concatenate(matrix_list)
            meta_list   = [x[1] for x in feature_df_list if x is not None]
            meta_df     = pd.concat(meta_list)
            parcel_list = feature_df_list[0][2]

            # write a single patientXfeature csv file for each combo of run/processing type/atlas
            uniq_ids = list(set(meta_df['uniq_id']))
            for uniq_id in uniq_ids:
                # subset
                cur_idxs  = np.where(meta_df['uniq_id'] == uniq_id)[0]
                write_df  = meta_df.loc[meta_df['uniq_id'] == uniq_id]
                write_mat = mat_stack[cur_idxs,:,:]
                # save to disk
                write_uniq_id = uniq_id.replace('.pconn.nii','')
                write_path    = Path(production_dir, f'{write_uniq_id}.hdf5')

                f = h5py.File(write_path,'w')
                dset = f.create_dataset('dataset', data=write_mat)
                dset = f.create_dataset('sample_data', data=write_df.to_numpy())
                dset = f.create_dataset('parcels', data=parcel_list)
                f.close()
                print(f'Saving to: {write_path}')


        # functional features and substrings to glob
        grep_tuples = (
            ('funccon/GBC', f'_atlas-{cifti_atlas_name}_stat-pearsonR_GBC.csv.gz'),
            ('funccon/GBC', f'_atlas-{cifti_atlas_name}_stat-pearsonRtoZ_GBC.csv.gz'),
            ('funccon/amplitude', f'_stat-fALFF_atlas-{cifti_atlas_name}.csv.gz'),
            #('funccon/amplitude', f'_stat-fRSFA_atlas-{cifti_atlas_name}.csv.gz'),
            ('funccon/amplitude', f'_stat-ALFF_atlas-{cifti_atlas_name}.csv.gz'),
            ('funccon/amplitude', f'_stat-RSFA_atlas-{cifti_atlas_name}.csv.gz'),
            #('funccon/amplitude', f'_stat-mALFF_atlas-{cifti_atlas_name}.csv.gz'),
            #('funccon/amplitude', f'_stat-mRSFA_atlas-{cifti_atlas_name}.csv.gz'),
            ('tsnr', f'_stat-TSNR_atlas-{cifti_atlas_name}.csv.gz'),
            ('task', f'-cope_study-EMBARC_atlas-{cifti_atlas_name}.csv.gz'),
            ('task', f'-zstat_study-EMBARC_atlas-{cifti_atlas_name}.csv.gz'),
            ('funccon/graphTheory', f'_atlas-{cifti_atlas_name}_stat-graphTheoryPearsonRtoZ.csv.gz')
        )
        for feature, glob_string in grep_tuples:
            print(f'Working on: {feature}')
            #glob_string = grep_dict[feature]
            feat_splits = feature.split('/')
            feature_name = feature.split('/')[-1]
            feature_folder = feature.split('/')[0]

            refined_dir = Path(output_dir, study_name, 'refined')
            # path to the output directory
            if len(feat_splits) == 2:
                production_dir = Path(output_dir, study_name, 'production', 'functional', feature_folder, feature_name)
            else: 
                production_dir = Path(output_dir, study_name, 'production', 'functional', feature_folder)

            production_dir.mkdir(exist_ok=True, parents=True)

            # parallel read all of the session directories
            with Pool(n_processes) as pool:
                parallelized_function = partial(
                    compile_production_data, 
                    refined_dir, feature, glob_string
                    )
                feature_df_list = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                                total=len(session_dirs)))
            feature_df = pd.concat(feature_df_list)
                                
            # write a single patientXfeature csv file for each combo of run/processing type/atlas
            uniq_ids = list(set(feature_df['uniq_id']))
            for uniq_id in uniq_ids:
                # subset
                write_df = feature_df.loc[feature_df['uniq_id'] == uniq_id]

                # save to disk
                write_uniq_id = uniq_id.replace('_long','')
                write_path = Path(production_dir, f'{write_uniq_id}.csv.gz')
                print(f'Saving to: {write_path}')
                write_df.to_csv(write_path, compression='gzip', index=None)


        # -------------------------------------
        # Compile CSV files with bad voxel info
        # -------------------------------------
        # functional features and substrings to glob
        refined_dir = Path(output_dir, study_name, 'refined')
        
        # path to the output directory
        production_dir = Path(output_dir, study_name, 'production', 'qc')
        production_dir.mkdir(exist_ok=True, parents=True)

        glob_string = f'_atlas-{cifti_atlas_name}_stat-BOLD_badvoxels.csv.gz'
        # parallel read all of the session directories
        with Pool(n_processes) as pool:
            parallelized_function = partial(
                compile_badvox_data, 
                refined_dir, glob_string
                )
            feature_df_list = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                            total=len(session_dirs)))
        long_badvox_df = pd.concat(feature_df_list)
        # save
        write_path = Path(production_dir, f'study-{study_name}_atlas-{cifti_atlas_name}_stat-BOLD_badvoxels.csv.gz')
        print(f'Saving to: {write_path}')
        long_badvox_df.to_csv(write_path, compression='gzip', index=None)


def compile_badvox_data(refined_dir, glob_string, session_dir):
    """
    TODO
    """
    try:
        session_id  = session_dir.stem
        session_refined_dir = Path(refined_dir, session_id, 'qc')
        csv_list = [x for x in Path(session_refined_dir).glob(f'*{glob_string}')]
        df_list  = [pd.read_csv(csv, compression='gzip') for csv in csv_list]
        session_df = pd.concat(df_list)
    except: 
        session_df = None
    return session_df


def surf_data_from_cifti(data, hdr_axis, surf_name):
    """
    Get data for a specific cifti "BRAIN_MODEL"
    """
    cii_axis = dlabel_cii.header.get_axis(1)
    for name, data_indices, model in cii_axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data

def dlabel_to_df(dlabel_path):
    """
    Read a dlabel.nii file and return a dataframe
    mapping parcel numbers and parcel names
    """
    dlabel_file = nb.load(dlabel_path)
    label_list  = [x for x in dlabel_file.header.get_axis(0)][0][1]
    label_dict  = {key:val[0] for key,val in label_list.items()}
    label_df    = pd.DataFrame({'roi_num':label_dict.keys(), 'roi':label_dict.values()})
    label_df    = label_df.loc[label_df['roi'] != '???']
    return label_df


def setup_logger(name, log_file):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)       
    formatter = logging.Formatter('%(asctime)s | %(message)s') 
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def process_session(
    study_output_dir,
    scans,
    input_scan_pattern,
    concatenate_scans,
    concatenate_scan_name,
    study_name,
    cifti_atlas,
    cifti_atlas_name,
    task_feat,
    session_dir
    ):
    """
    Produce BOLD features from a given QuNex session directory

    Specifically this script will calculate:
        1. Resting-State Functional Connectivity
        2. Global Brain Connectivity
        3. Amplitude Measures (fALFF, ALFF, RSFA)
        4. Resting-State Functional Covariance
        5. Graph Theory Metrics
        6. [TODO] Regional Homogeneity

    Parameters
    ----------
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    output_folder: str
        Where to save the outputs of this script
        Example: /fmri-qunex/research/imaging/datasets/embarc/qunex_features
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
        Example: '(session_a|session_b|session_c)' to capture only
            'session_a', 'session_b', and 'session_c' sessions
    input_scan_pattern: str
        String to identify the QuNex output scan for processing. 
        It is recommended to included the ".dtseries.nii" to avoid
        potentially grabbing multiple filetypes. 
        Example: '_Atlas_s_hpss_res-mVWMWB_lpss.dtseries.nii'
    concatenate_scans: bool
        Should the scans be concatenated (in time)
        Example: True
    concatenate_scan_name: str
        Name of the newly produced concatenated scan name
    study_name: str
        A descriptive name of the imaging project/QuNex output being processed
        Example: EMBARC
    cifti_atlas: str
        [Optional] Path to the CIFTI dlabel.nii file in fslr32k HCP space
        Example: ~/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
    cifti_atlas_name: str
        [Optional] Descriptive name for the CIFTI dlabel.nii file
        Example: CABNP
    n_processes: int
        Number of CPUs to use for parallel computation
        Example: 2
    """
    try: 

        # Session ID
        session_id = session_dir.stem
        
        # Output Directories
        raw_out_dir      = utilities.make_directory(Path(study_output_dir, 'raw', session_id, 'functional'))
        raw_func_dir     = utilities.make_directory(Path(study_output_dir, 'raw', session_id, 'functional'))
        refined_func_dir = utilities.make_directory(Path(study_output_dir, 'refined', session_id, 'functional', 'funccon'))


        # "raw" functional output directory
        ##raw_out_dir = Path(study_output_dir, 'raw', session_id, 'functional')
        ##raw_out_dir.mkdir(exist_ok=True, parents=True)
        # "refined" output directory
        ##refined_func_dir = Path(study_output_dir, 'refined', session_id, 'functional', 'funccon')
        ##refined_func_dir.mkdir(exist_ok=True, parents=True)

        # set up session specific logger
        dt      = datetime.now().strftime('%Y%m%d_%Hh_%Mm_%Ss')
        log_dir = Path(study_output_dir, 'logs')
        log_dir.mkdir(exist_ok=True, parents=True)
        session_log = Path(log_dir, f'{session_id}_REST_runlog_{dt}.log')
        logger = setup_logger(session_id, session_log)
        logger.info('------ INPUT ARGUMENTS ------')
        logger.info(f'FUNCTION: {inspect.stack()[0][3]}')

        # log input arguments
        argspec = inspect.getargvalues(inspect.currentframe())
        [logger.info(f'{k} = {argspec.locals[k]}') for k in argspec.locals.keys()]
        logger.info('\n\n')

        # Get scan paths, stored as list of tuples
        # (bold_number, bold_long_name, bold_full_path)
        # --------------
        orig_scan_tuples = neuroimage.get_dtseries_paths(scans, input_scan_pattern, session_dir)

        #func_get_dtseries = Function(input_names=["scans", "input_scan_pattern", "session_dir"],
        #                             output_names=["out_tuples"],
        #                             function=get_dtseries_paths)
        #func_get_dtseries.inputs.scans = scans
        #func_get_dtseries.inputs.input_scan_pattern = input_scan_pattern
        #func_get_dtseries.inputs.session_dir = session_dir

        # fALFF/ALFF calculations require non-bandpassed dtseries
        # i.e. the file without the "lpss" flag
        falff_scan_pattern = input_scan_pattern.replace('_lpss','')
        falff_scan_tuples  = get_dtseries_paths(scans, falff_scan_pattern, session_dir)


        # --------------
        # Process Task contrasts
        # --------------
        if task_feat != (None,):
            stat_files = []
            refined_task_dir = Path(study_output_dir, 'refined', session_id, 'functional', 'task')
            refined_task_dir.mkdir(exist_ok=True)                
            for task_feat_in in task_feat: 
                feat_name, feat_dir = task_feat_in.split('|')

                feat_tuples = get_task_contrasts(session_dir, feat_dir)
                if feat_tuples == None:
                    continue 
                for feat_tuple in feat_tuples:
                    contr_num, contr_name, scan_path_c, scan_path_z = feat_tuple
                    
                    # Copy COPE file
                    new_fname = f'sub-{session_id}_task-{feat_name}_contrast-{contr_name}_stat-cope_study-{study_name}.dtseries.nii'
                    new_cope  = Path(refined_task_dir, new_fname)
                    # do the copy
                    if new_cope.exists():
                        os.remove(new_cope)
                    shutil.copyfile(scan_path_c, new_cope)
                        
                    # Copy ZSTAT file
                    new_fname = f'sub-{session_id}_task-{feat_name}_contrast-{contr_name}_stat-zstat_study-{study_name}.dtseries.nii'
                    new_zstat = Path(refined_task_dir, new_fname)
                    # do the copy
                    if new_zstat.exists():
                        os.remove(new_zstat)
                    shutil.copyfile(scan_path_z, new_zstat)

                    stat_files.append(new_cope)
                    stat_files.append(new_zstat)

            for dtseries in stat_files:
                ptseries = str(dtseries).replace('.dtseries', f'_atlas-{cifti_atlas_name}.ptseries')
                cifti_parcellate(dense_in=dtseries, 
                            cifti_atlas=cifti_atlas, 
                            parcel_out=ptseries,
                            logger=logger)

                # convert pscalar >> csv.gz
                pscalar_in  = nb.load(ptseries)
                parcel_list = [p.name for p in pscalar_in.header.matrix[1].parcels]
                values      = pscalar_in.dataobj[0]
                contr_df = pd.DataFrame(values).transpose()
                contr_df.columns = parcel_list 
                contr_df.insert(0, 'session_id', session_id)
                contr_df.insert(1, 'src_file', ptseries.split('/')[-1])
                csv_out = str(ptseries).replace('.ptseries.nii', '.csv.gz')
                contr_df.to_csv(csv_out, compression='gzip', index=None)
                # average Amplitude measure by network
                #summarize_feat_by_network(csv_file=Path(csv_out), feat_dim='1D')


        # --------------
        # Copy
        # --------------
        # copy and rename original dtseries to the raw/functional output dir
        # keys = bold_number
        # values = full_path_to_orig_dtseries
        logger.info('------ COPY DTSERIES ------')
        copy_cii_dict = {}
        for scan_num, scan_name, scan_path in orig_scan_tuples:
            inputs = [session_id, scan_num, scan_name, scan_path, raw_out_dir, logger]
            copy_cii_dict[scan_num] = copy_dtseries_files(study_name, 
                                                            *inputs)
        
        # same, but for files needed for falff calculation
        falff_copy_cii_dict = {}
        for scan_num, scan_name, scan_path in falff_scan_tuples:
            inputs = [session_id, scan_num, scan_name, scan_path, raw_out_dir, logger]
            falff_copy_cii_dict[scan_num] = copy_dtseries_files(study_name, 
                                                            *inputs)


        # --------------
        # Censor Motion Frames
        # --------------
        logger.info('------ MOTION CENSOR ------')
        motion_cii_dict = {}
        for scan_num, scan_path in copy_cii_dict.items():
            inputs = [scan_num, scan_path, logger]
            motion_cii_dict[scan_num] = motion_censor(session_dir, 
                                                        scan_num, 
                                                        scan_path, 
                                                        logger)
        
        falff_motion_cii_dict = {}
        for scan_num, scan_path in falff_copy_cii_dict.items():
            inputs = [scan_num, scan_path, logger]
            falff_motion_cii_dict[scan_num] = motion_censor(session_dir, 
                                                        *inputs)

        # --------------
        # Quantify BOLD spatial coverage for each atlas parcel
        # --------------
        refined_qc_dir = Path(study_output_dir, 'refined', session_id, 'qc')
        refined_qc_dir.mkdir(exist_ok=True)
        for scan_num, scan_name, scan_path in orig_scan_tuples:
            # Path to folder containing "goodvox" data
            sub_dir    = f'MNINonLinear/Results/{scan_num}/RibbonVolumeToSurfaceMapping'
            ribbon_dir = Path(session_dir, 'hcp', session_id, sub_dir)

            # quantify the number/percentage of bad voxels within each parcel
            voxel_coverage_df = bad_voxels_by_parcel(ribbon_dir, cifti_atlas)

            # format/add scan info to the badvoxel dataframe
            voxel_coverage_df = voxel_coverage_df.drop(columns='labels')
            voxel_coverage_df.insert(0, 'session_id', session_id)
            voxel_coverage_df['scan_num'] = scan_num
            voxel_coverage_df['scan_name'] = scan_name

            # save 
            clean_scanname = scan_name.replace('bold ', '').replace(' ','_').replace('-','')
            out_fname = f'sub-{session_id}_task-{clean_scanname}_study-{study_name}_atlas-{cifti_atlas_name}_stat-BOLD_badvoxels.csv.gz'
            out_path  = Path(refined_qc_dir, out_fname)
            voxel_coverage_df.to_csv(out_path, index=None, compression='gzip')


        # -----------------
        # Concatenate
        # -----------------
        if concatenate_scans == True:
            logger.info('------ CONCAT CIFTIS ------')
            # list of ciftis to merge 
            cii_list  = list(motion_cii_dict.values())
            # only do this if we've found more than one scan to concat
            if len(cii_list) > 1:
                inputs = [cii_list, concatenate_scan_name, raw_out_dir, logger]
                concat_cii = concatenate_ciftis(*inputs)
                motion_cii_dict['concat'] = concat_cii

            falff_cii_list  = list(falff_motion_cii_dict.values())
            if len(falff_cii_list) > 1:
                inputs = [falff_cii_list, concatenate_scan_name, raw_out_dir, logger]
                falff_concat_cii = concatenate_ciftis(*inputs)
                falff_motion_cii_dict['concat'] = falff_concat_cii


        # -----------------
        # Parcellate
        # -----------------
        logger.info('------ PARCELLATE CIFTI ------')
        ptseries_list = []
        for dtseries in motion_cii_dict.values():
            ptseries = str(dtseries).replace('.dtseries.nii', f'_atlas-{cifti_atlas_name}.ptseries.nii')
            # wrapper for wb_command
            cifti_parcellate(dense_in=dtseries, 
                                cifti_atlas=cifti_atlas, 
                                parcel_out=ptseries,
                                logger=logger)
            ptseries_list.append(ptseries)

        # -------------------
        # (f)ALFF, ALFF, RSFA
        # -------------------
        logger.info('------ (f)ALFF ------')
        amp_list = []
        for scan_name, scan_path in falff_motion_cii_dict.items():
            amp_files = cii_falff_wrapper(scan_path, raw_out_dir)
            amp_list += [d for d in amp_files if d.exists()]

        # -----------------
        # (f)ALFF, LFF, RSFA - Parcellate
        # -----------------
        # TODO: put the messy csv output into a function
        amp_refined_out = Path(refined_func_dir, 'amplitude')
        amp_refined_out.mkdir(exist_ok=True, parents=True)
        for dscalar in amp_list:
            pscalar = Path(amp_refined_out, dscalar.name.replace('.dscalar.nii', f'_atlas-{cifti_atlas_name}.pscalar.nii'))
            cifti_parcellate(dense_in=dscalar, 
                                cifti_atlas=cifti_atlas, 
                                parcel_out=pscalar,
                                logger=logger)
            # convert pscalar >> csv.gz
            pscalar_in = nb.load(pscalar)
            parcel_list = [p.name for p in pscalar_in.header.matrix[1].parcels]
            values = pscalar_in.dataobj[0]
            amp_df = pd.DataFrame(values).transpose()
            amp_df.columns = parcel_list 
            amp_df.insert(0, 'session_id', session_id)
            amp_df.insert(1, 'src_file', pscalar.name)
            csv_out = str(pscalar).replace('.pscalar.nii', '.csv.gz')
            amp_df.to_csv(csv_out, compression='gzip', index=None)
            # average Amplitude measure by network
            summarize_feat_by_network(csv_file=Path(csv_out), feat_dim='1D')

        # --------------
        # BOLD stats
        # ---------------
        for cii_file in ptseries_list: 
            logger.info(cii_file)
            # Pearson R
            logger.info('------ Pearson R ------')
            r_pconn, r_csv, r_csv_long, r_csv_wide = cifti_correlate(cii_file,
                                                            out_dir=refined_func_dir,
                                                            stat_name='pearsonR',
                                                            fisher=False,
                                                            covariance=False)
            summarize_feat_by_network(csv_file=r_csv_long, feat_dim='2D')

            # Global Brain Connectivity (from Pearson R)
            logger.info('------ Global Brain Connectivity (Pearson) ------')
            gbc_file = compute_gbc(conn_file=r_pconn, stat_name='pearsonR_GBC', out_dir=refined_func_dir)
            summarize_feat_by_network(csv_file=gbc_file, feat_dim='1D')


            # Pearson R (Fisher transform)
            logger.info('------ Pearson R to Z ------')
            rz_pconn, rz_csv, rz_csv_long, rz_csv_wide = cifti_correlate(cii_file,
                                                            out_dir=refined_func_dir,
                                                            stat_name='pearsonRtoZ',
                                                            fisher=True,
                                                            covariance=False)
            summarize_feat_by_network(csv_file=rz_csv_long, feat_dim='2D')
            logger.info('------ Graph Theory ------')
            graph_theory_metrics(rz_csv, session_id, stat_name='graphTheoryPearsonRtoZ')

            # Global Brain Connectivity (from Pearson RtoZ)
            logger.info('------ Global Brain Connectivity (Pearson R to Z) ------')
            gbc_r2z_file = compute_gbc(conn_file=rz_pconn, stat_name='pearsonRtoZ_GBC', out_dir=refined_func_dir)
            summarize_feat_by_network(csv_file=gbc_r2z_file, feat_dim='1D')

            # Covariance
            logger.info('------ Covariance ------')
            cov_pconn, cov_csv, cov_csv_long, cov_csv_wide = cifti_correlate(cii_file,
                                                            out_dir=refined_func_dir,
                                                            stat_name='covariance',
                                                            fisher=False,
                                                            covariance=True)
            summarize_feat_by_network(csv_file=cov_csv_long, feat_dim='2D')

        # remove motion censor dtseries to save space
        for key,cii_path in motion_cii_dict.items():
            if Path(cii_path).exists():
                os.remove(str(cii_path))

        # -----------------
        # TSNR parcellation
        # -----------------
        logger.info('------ PARCELLATE TSNR ------')
        refined_tsnr_dir = Path(study_output_dir, 'refined', session_id, 'functional', 'tsnr')
        refined_tsnr_dir.mkdir(exist_ok=True)

        session_path = Path(session_dir, 'session_hcp.txt')
        scan_dict    = parse_session_hcp(session_path, scans)
        for scan_name, scan_num in scan_dict.items():
            try:
                scan_name          = scan_name.replace('bold ', '').replace(' ','').replace('-','_')
                scan_dir           = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/Results', scan_num)
                tsnr_file          = Path(scan_dir, '{}_Atlas_TSNR.dscalar.nii'.format(scan_num))
                out_fname          = f'sub-{session_dir.stem}_task-{scan_name}_proc-Atlas_study-{study_name}_stat-TSNR_atlas-{cifti_atlas_name}.pscalar.nii'
                tsnr_parcel_output = Path(refined_tsnr_dir, out_fname)
                cifti_parcellate(dense_in=tsnr_file, 
                                    cifti_atlas=cifti_atlas, 
                                    parcel_out=tsnr_parcel_output,
                                    logger=logger)

                # convert pscalar >> csv.gz
                pscalar_in  = nb.load(tsnr_parcel_output)
                parcel_list = [p.name for p in pscalar_in.header.matrix[1].parcels]
                values  = pscalar_in.dataobj[0]
                tsnr_df = pd.DataFrame(values).transpose()
                tsnr_df.columns = parcel_list 
                tsnr_df.insert(0, 'session_id', session_id)
                tsnr_df.insert(1, 'src_file', tsnr_file.name)
                csv_out = str(tsnr_parcel_output).replace('.pscalar.nii', '.csv.gz')
                tsnr_df.to_csv(csv_out, compression='gzip', index=None)
                # average Amplitude measure by network
            except: 
                continue 
    except Exception as e:
        print(e) 


def get_task_contrasts(session_dir, feat_dir):
    """
    TODO
    """
    session_id = session_dir.stem

    # get full feat directory
    results_dir = Path(session_dir, 'hcp', session_id, 'MNINonLinear/Results')
    feat_list   = list(results_dir.glob(f'*/{feat_dir}'))
    try:
        if len(feat_list) != 1:
            raise ValueError(f'Feat Directory not found.\n{results_dir}\n{feat_dir}')
        else: 
            full_feat_dir = feat_list[0]

        # Read design contrast
        design_con_file = Path(full_feat_dir, 'design.con')
        with open(str(design_con_file), 'r') as f:
            design_con = f.readlines()
        
        # create dictionary describing each individual first-level contrast
        contrast_list  = [x for x in design_con if 'ContrastName' in x]
        contrast_nums  = [x.split('Name')[1].split('\t')[0] for x in contrast_list]
        contrast_nums  = [x.split('Name')[1].split('\t')[0] for x in contrast_list]
        contrast_names = [x.split('\t')[1].replace('\n','').replace(' ','') for x in contrast_list]
        contrast_dict  = dict(zip(contrast_nums, contrast_names))

        # read contrast COPE files
        contrast_num = '1'
        contrast_tuples = []
        for contrast_num in contrast_dict.keys():
            cope_file  = Path(full_feat_dir, 'GrayordinatesStats', f'cope{contrast_num}.dtseries.nii')
            zstat_file = Path(full_feat_dir, 'GrayordinatesStats', f'zstat{contrast_num}.dtseries.nii')
            if cope_file.exists() and zstat_file.exists():
                return_tuple = (contrast_num, contrast_dict[contrast_num], cope_file, zstat_file)
                contrast_tuples.append(return_tuple)
        return contrast_tuples
    except Exception as e:
        print(e)
    

if __name__ == '__main__':
    main()