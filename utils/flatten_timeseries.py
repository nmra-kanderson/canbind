import gzip
import pathlib
import re
import traceback
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List

import click
import boto3
import numpy as np
import tqdm

from utils.neuroimage import load
from utils.neuroimage import load_motion_scrub
from utils.neuroimage import parse_session_hcp


def parse_scan(
        scan_types: List[str],
        scan: str,
        ftype: str,
        delimiter: str,
        output_folder: str,
        overwrite: bool,
        s3_destination_bucket: str,
        s3_destination_path: str,
        session_path: pathlib.PosixPath
):
    """
    Load dense timeseries (.dtseries.nii) files, flatten, and include motion
    scrub info. Saves the file locally or optionally directly to S3. If you
    specify S3 details the function assumes that you don't need the local
    copy after the transfer is complete and will delete the file to save
    space.

    Parameters
    ----------
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
    scan: str
        The specific scan that is being processed in this function call
        Example: 'bold reward'
    ftype: str
        The suffix of the .nii file.
        Example: _Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0.dtseries.nii
            in the file bold1_Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0.dtseries.nii
    delimiter: str
        How to separate values on each line in the output file
        Example: ',' for CSV, \t for tab
    output_folder: str
        Where to save the outputs of this script
        Example: ~/embarc/flat_files
    overwrite: bool
        Whether to overwrite output files if they exist
    s3_destination_bucket: str
        The S3 bucket to save the file to. This was helpful for
        HCP where saving the intermediate files on S3 made FSX
        run out of space. It was better to save the files to S3
        directly. Example: 'neuro-trestle-ds'
    s3_destination_path: str
        Where to save the file on S3. Should not contain the
        name of the file itself. The code assumes that the file
        name that exists locally should be the same as the
        file name on S3. Example:
            'trestle/data_sources/hcp/fmri/processed_data/tseries'
    session_path: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline

    Notes
    -----
    Overwrite the file if it exists from a previous run to avoid accidental duplicates

    """

    try:
        # Build the mapping from rest/task to the bold scan, e.g., bold1, bold2, etc.
        session_hcp_path = Path(session_path).joinpath('session_hcp.txt')
        mapping = parse_session_hcp(file_path=session_hcp_path, scan_types=scan_types)

        # Use the mapping to find the prefix (e.g., bold2) for this scan (e.g., rest-1)
        file_prefix = 'bold' + mapping[scan]

        # Get path to actual dtseries files
        hcp_path = session_path.joinpath('images', 'functional')
        scan_file_path = hcp_path.joinpath(file_prefix + ftype)

        # Extract data from file into a numpy array of shape (timepoints, regions)
        dtseries = load(str(scan_file_path))
        ntimepoint, nvoxel = dtseries.shape

        # build header
        header = ['session_id', 'scan', 'frame', 'use']
        voxel_names = [f'voxel_{n}' for n in np.arange(nvoxel).astype(str)]
        header += voxel_names

        # create filename
        session_id = Path(session_path).stem
        scan_fixed = scan.replace('bold ', '').replace(' ', '_').replace('-', '_')
        suffix = 'txt.gz'
        if delimiter == ',':
            suffix = 'csv.gz'
        elif delimiter == '\t':
            suffix = 'tsv.gz'
        output_file_name = session_id + "-" + scan_file_path.name.replace(
            file_prefix + "_", scan_fixed + "-").replace("nii", suffix)
        output_file = pathlib.Path(output_folder).joinpath(output_file_name)

        # determine motion scrubbed frames
        use = load_motion_scrub(
            session_path=session_path, file_prefix=file_prefix)
        assert len(use) == ntimepoint
        assert -2 not in use

        # Optionally the delete file if it already exists
        if overwrite and Path(output_file).exists():
            # https://docs.python.org/3/library/pathlib.html#pathlib.Path.unlink
            output_file.unlink()

        # write to file. overwrite if exists
        with gzip.open(output_file, 'wb') as file_writer:
            header_line = delimiter.join(header) + '\n'
            file_writer.write(header_line.encode())
            for frame, use_frame in enumerate(use):
                line = delimiter.join(
                    [str(session_id), scan_fixed, str(frame), str(use_frame)]
                ) + delimiter
                line += delimiter.join(dtseries[frame, :].astype(str)) + '\n'
                file_writer.write(line.encode())

        if s3_destination_bucket and s3_destination_path:

            # Use the AWS resource's IAM role or laptop's default aws profile
            s3 = boto3.resource('s3')

            # Remove trailing and leading slashes so S3 doesn't complain
            s3_destination_path = s3_destination_path.strip('/')

            # Copy the file to S3
            s3.Bucket(s3_destination_bucket).upload_file(
                f'{output_folder}/{output_file_name}',
                f'{s3_destination_path}/{output_file_name}')

            # Remove the local file to save space
            output_file.unlink()

    except:
        print(f'Could not process {scan} for session: {session_path}')
        traceback.print_exc()


@click.command()
@click.option('--scans', multiple=True)
@click.option('--ftype')
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-folder')
@click.option('--n-processes', type=int, default=4)
@click.option('--delimiter', default=',')
@click.option('--regex-session-filter', default=None)
@click.option('--overwrite', is_flag=True, default=False)
@click.option('--s3-destination-bucket', default=None)
@click.option('--s3-destination-path', default=None)
def main(
        scans: List[str],
        ftype: str,
        sessions_dir: pathlib.PosixPath,
        output_folder: str,
        n_processes: int,
        delimiter: str,
        regex_session_filter: str,
        overwrite: bool,
        s3_destination_bucket: str,
        s3_destination_path: str
):
    """
    Convert BOLD .nii (cifti) timeseries data to text files.
    Each .nii file will get its own text file. The resulting
    timeseries will not include "scrubbed" frames (i.e.
    timepoints with above-threshold motion).

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
    ftype: str
        The suffix of the .nii file.
        Example: _Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0.ptseries.nii
            in the file bold1_Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0.ptseries.nii
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    output_folder: str
        Where to save the outputs of this script
        Example: ~/embarc/flat_files
    n_processes: int
        The number of parallel processes to use. Makes the code
        run a lot faster (minutes vs hours)
        Example: 20
    delimiter: str
        How to separate values on each line in the output file
        Example: ',' for CSV, \t for tab
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
        Example: '(session_a|session_b|session_c)' to capture only
            'session_a', 'session_b', and 'session_c' sessions
    overwrite: bool
        Whether to overwrite output files if they exist. This can
        be helpful if you have to do a partial re-run where some
        of the output files got corrupted and you want to get rid
        of the bad data first. It's often helpful to use this
        flag along with a --regex-session-filter CLI arg of pipe
        delimited sessions like '(session_a|session_b|session_c)'
        so that you don't end up re-running and overwriting
        good sessions too.
    s3_destination_bucket: str
        The S3 bucket to save the file to. This was helpful for
        HCP where saving the intermediate files on S3 made FSX
        run out of space. It was better to save the files to S3
        directly. Example: 'neuro-trestle-ds'
    s3_destination_path: str
        Where to save the file on S3. Should not contain the
        name of the file itself. The code assumes that the file
        name that exists locally should be the same as the
        file name on S3. Example:
            'trestle/data_sources/hcp/fmri/processed_data/tseries'
    """

    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Identify all session directories
    session_dirs = list()
    for path in Path(sessions_dir).iterdir():
        if regex_session_filter:
            if re.match(regex_session_filter, path.stem):
                session_dirs.append(path)
    n_sessions = len(session_dirs)

    print(f'\nFound {n_sessions} sessions\n')

    # Process each of the scan types
    for scan in scans:
        # Parallelize CPU-intensive workloads for a big speed boost
        with Pool(n_processes) as pool:
            """
            Due to how ProcessPoolExecutor is designed, you can't pass the
            parallelized function multiple function arguments unless you
            wrap the function and its arguments in `partial`
            """
            parallelized_function = partial(
                parse_scan, scans, scan, ftype, delimiter, output_folder,
                overwrite, s3_destination_bucket, s3_destination_path)

            # Process results in parallel, and show a progress bar via tqdm
            print(f'\nProcessing scans of type: {scan}')
            _ = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                               total=len(session_dirs)))


if __name__ == '__main__':
    main()
