import os
import glob
import re
from pathlib import Path
import pathlib
import traceback
import click
import subprocess
import pandas as pd
import nibabel as nb
from multiprocessing import Pool
from functools import partial
import tqdm
from typing import List

from utils.neuroimage import parse_session_hcp, load_motion_scrub, get_qunex_dirs, cifti_parcellate


@click.command()
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-folder')
@click.option('--cifti-atlas', default=None)
@click.option('--cifti-atlas-name', default=None)
@click.option('--regex-session-filter', default=None)
@click.option('--study-name', default=None)
@click.option('--input-file-pattern', default=None, multiple=True)
@click.option('--n-processes', type=int, default=4)
#@click.option('--input-file-pattern', multiple=True)

def main(
        sessions_dir: pathlib.PosixPath,
        regex_session_filter: str,
        output_folder: str,
        study_name: str,
        cifti_atlas_name: str,
        cifti_atlas: str,
        n_processes: int,
        input_file_pattern: List[str],
        #input_file_pattern: str
        ):
    """
    Produce a parcellated timeseries (*ptseries.nii) from dense timeseries 
    (i.e. *.dtseries.nii). Assumes QuNex style aves the files locally. 

    Parameters
    ----------
    session_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline
    output_folder: pathlib.Path
        Path to the output directory 
        Example: /fmri-qunex/research/imaging/datasets/embarc/qunex_features
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
    cifti_atlas: str
        Full path to the 92k volume/surface CIFTI atlas (i.e. *.dlabel.nii)
        e.g. ${repo}/reference/parcellation/ColeAnticevic/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
    cifti_atlas_name: str
        Descriptive name or acronym for the CIFTI atlas. 
        e.g. CABNP
    scan_types: list[str]
        The names of scans to process. The names must exactly match
        what appears in the session_hcp.txt file. Not all scan types
        need to appear in all of the session_hcp.txt files from every
        session, but "ert" won't pick up "bold ert", if "bold ert"
        is what appears in session_hcp.txt. Within a given study,
        some sessions will assign different IDs to the same scan types
        depending on scan availability within a session. QuNex doesn't
        like gaps in BOLD ID numbers within a session
        Example: ['bold ert', 'bold rest run-1', 'bold rest run-2',
            'bold reward']
    study_name: str
        Descriptive name or acryonym for the study
        e.g. EMBARC
    n_processes: int
        The number of parallel processes to use. Makes the code
        run a lot faster (minutes vs hours)
        Example: 20
    input_file_pattern: list[str]
        A pattern used to identify a session's input file that you
        would like to concatenate with other input files
        For example if you wanted to concatenate:
            UM0121_baseline-rest_run_1-Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CABNP.ptseries.csv.gz
            UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CABNP.ptseries.csv.gz
        you would specify:
            --input-file-pattern rest_run_1-Atlas_s_hpss_res-mVWMWB_lpss_BOLD \
            --input-file-pattern rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss_BOLD
        in the CLI args. The order of series in the final concatenated
        series will match the order of series mentioned in the CLI args,
        so 1 before 2 in the example above.
    """
    print('Sessions Dir:     {}'.format(str(sessions_dir)))
    print('Output Dir:       {}'.format(str(output_folder)))
    print('Regex Filter:     {}'.format(str(regex_session_filter)))
    print('Study Name:       {}'.format(str(study_name)))
    print('CIFTI Atlas:      {}'.format(str(cifti_atlas)))
    print('CIFTI Atlas Name: {}'.format(str(cifti_atlas_name)))
    print('Scan Types:       {}'.format(input_file_pattern))
    print('N Processes:      {}'.format(n_processes))

    # Create output folder if it doesn't exist
    output_folder       = Path(output_folder)
    study_output_folder = Path(output_folder, study_name)
    output_folder.mkdir(parents=True, exist_ok=True)
    study_output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of QuNex subdirectories for processing
    session_dirs = get_qunex_dirs(sessions_dir, regex_session_filter)
    session_dir = session_dirs[0]

    scan_pattern = input_file_pattern[0]
    for scan in input_file_pattern:
        print(f'Processing Scan: {scan}')

        # Parallelize workloads for a big speed boost
        with Pool(n_processes) as pool:
            parallelized_function = partial(
                parcellate_session,
                input_file_pattern, study_output_folder, scan, study_name, cifti_atlas, cifti_atlas_name)

            # Process results in parallel, and show a progress bar via tqdm
            _ = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                               total=len(session_dirs)))



def parcellate_session(input_file_pattern, study_output_folder, scan_pattern, study_name, cifti_atlas, cifti_atlas_name, session_dir):
    """
    Produce a parcellated timeseries (*ptseries.nii) from dense timeseries 
    (i.e. *.dtseries.nii). Assumes QuNex style aves the files locally. 

    Parameters
    ----------
    session_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline
    output_folder: pathlib.Path
        Path to the output directory 
        Example: /fmri-qunex/research/imaging/datasets/embarc/qunex_features
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
    cifti_atlas: str
        Full path to the 92k volume/surface CIFTI atlas (i.e. *.dlabel.nii)
        e.g. ${repo}/reference/parcellation/ColeAnticevic/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
    cifti_atlas_name: str
        Descriptive name or acronym for the CIFTI atlas. 
        e.g. CABNP
    scan_types: list[str]
        The names of scans to process. The names must exactly match
        what appears in the session_hcp.txt file. Not all scan types
        need to appear in all of the session_hcp.txt files from every
        session, but "ert" won't pick up "bold ert", if "bold ert"
        is what appears in session_hcp.txt. Within a given study,
        some sessions will assign different IDs to the same scan types
        depending on scan availability within a session. QuNex doesn't
        like gaps in BOLD ID numbers within a session
        Example: ['bold ert', 'bold rest run-1', 'bold rest run-2',
            'bold reward']
    study_name: str
        Descriptive name or acryonym for the study
        e.g. EMBARC
    n_processes: int
        The number of parallel processes to use. Makes the code
        run a lot faster (minutes vs hours)
        Example: 20
    """

    # create subjects specific output directory in the {}/raw subfolder
    session_id = session_dir.stem
    session_output_folder = Path(study_output_folder, 'raw', session_id, 'functional')
    session_output_folder.mkdir(exist_ok=True, parents=True)

    # find matches
    cii_matches = [cii for cii in session_dir.rglob(f'*{scan_pattern}')]

    for dtseries_file in cii_matches:
        ptseries_path = str(dtseries_file).replace('.dtseries.nii', f'_atlas-{cifti_atlas_name}.ptseries.nii')
        # wrapper for wb_command
        cifti_parcellate(dense_in=dtseries_file, 
                            cifti_atlas=cifti_atlas, 
                            parcel_out=ptseries_path)

if __name__ == '__main__':
    main()
