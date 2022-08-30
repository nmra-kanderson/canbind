import os
import re
import inspect
import logging
import itertools
import subprocess
from distutils.version import StrictVersion
from pathlib import Path
from typing import List
from scipy.fftpack import fft

import nibabel as nib
import nibabel as nb
import numpy as np
import pandas as pd

import reference.parcellation
from utilities import *
from constants import QUILT_REF_PACKAGE,TRESTLE_REGISTRY,MRI_CONVERT,MRI_ANATOMICAL_STATS,FREESURFER_HOME,WB_COMMAND,DRSFC,GRAPH_METRICS_SCRIPT


__all__ = [
            'parse_session_hcp',
            'stack_pconn_matrices',
            'get_qunex_dirs',
            'cifti_parcellate',
            'stack_pconn_matrices',
            'get_dtseries_paths',
            'copy_dtseries_files',
            'motion_censor',
            'bad_voxels_by_parcel',
            'concatenate_ciftis',
            'cii_falff_wrapper',
            'summarize_feat_by_network',
            'cifti_correlate',
            'graph_theory_metrics',
            'compute_gbc'
            ]

#__all__ = ['load', 'parse_session_hcp', 'parse_session_hcp', 'scrub_motion',
#           'load_motion_scrub', 'parcellation_from_file_name',
#           'load_parcellation_mapping', 'session_id_from_file_name',
#           'tseries_unit_from_file_name', 'file_match', 'print_feedback',
#           'load_expected_parcel_order', 'valid_parcel_schema',
#           'enforce_parcel_order', 'scan_type_from_file_name']

if StrictVersion(nib.__version__) < StrictVersion('3.0.0'):
    raise EnvironmentError('update nibabel to version 3')


def print_feedback(print_string):
    """
    Print fancy feedback
    """
    print('------------------------------')
    print(print_string)
    print('------------------------------')


def enforce_parcel_order(parcellation_method, data_frame):
    """
    Ensures that parcels always appear in the expected order

    Parameters
    ----------
    nonparcel_columns : List[str]
        A list of the columns unrelated to parcels. These
        should always appear first in the file. Example: [
            'scan', 'frame', 'use', 'parcel_A', 'parcel_B']
    parcel_columns: List[str]
        The expected order of parcels. Example: [
            'parcel_A', 'parcel_B', 'parcel_C']
    data_frame: pandas.DataFrame
        All of the data in question

    Returns
    -------
    data_frame: pandas.DataFrame
        The data frame with correctly ordered parcel
        columns
    """

    # Preserve order of non-parcel columns
    parcel_names_expected_order = load_expected_parcel_order(parcellation_method)
    nonparcel_columns = list(data_frame.columns)
    for parcel_name in parcel_names_expected_order:
        if parcel_name in nonparcel_columns:
            nonparcel_columns.remove(parcel_name)
    column_order = nonparcel_columns + parcel_names_expected_order
    return data_frame.reindex(columns=column_order)


def valid_parcel_schema(parcel_schema, file_header):
    """
    Checks that all of the parcels exist and appear in the
    correct order in the given file header

    Parameters
    ----------
    parcel_schema : List[str]
        A list of parcel names from a reference schema.
        Reference schemas are saved in the repo in the
        reference/parcellation/mapping/ folder
        Example: ['Parcel_A', 'Parcel_B', 'Parcel_C']
    file_header: List[str]
        A header from a CSV file. Example: [
            'scan', 'frame', 'use', 'parcel_A', 'parcel_B']

    Returns
    -------
    is_valid: bool
    """
    try:
        if type(parcel_schema) != List[str]:
            parcel_schema = list(parcel_schema)
        if type(file_header) != List[str]:
            file_header = list(file_header)
        if parcel_schema[0] in file_header:
            start = file_header.index(parcel_schema[0])
            header_parcels = file_header[start:start + len(parcel_schema)]
            return header_parcels == parcel_schema
        else:
            return False
    except:
        return False


def file_match(file_name, tseries_unit, other_pattern, parcellation_method=None):
    """
    Checks whether a file meets all of the given match
    criteria. Helpful when iterating over a wide variety
    of files in a directory. For example, when there are
    different parcellation methods, the folder contains
    dtseries as well as ptseries, etc.

    Parameters
    ----------
    file_name : str
        Example: UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss-CABNP.ntseries.csv.gz
    tseries_unit: str
        Whether the data involves voxels / greyordinates,
        parcels, or networks. Example: 'd', 'p', or 'n'
    other_pattern: str
        Anything else you might use to filter a file
        that is not already captured by explicit args
    parcellation_method: str
        Parcellation method, if applicable, e.g., 'CABNP'

    Returns
    -------
    is_match: bool
    """
    if tseries_unit in ['p', 'n']:
        return tseries_unit_from_file_name(file_name) == tseries_unit and \
            parcellation_from_file_name(file_name) == parcellation_method and \
            other_pattern in file_name
    else:
        return parcellation_from_file_name(file_name) == parcellation_method


def parcellation_from_file_name(file_name):
    """
    Gets the parcellation method from a *tseries file
    name. This is a simple operation, but someday the
    file pattern might change. If it does, only this
    function will need to be updated instead of all the
    functions that rely on it

    Parameters
    ----------
    file_name : str
        Example: UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss-CABNP.ntseries.csv.gz

    Returns
    -------
    parcellation: str
        Example: 'CABNP'
    """
    try:
        return file_name.split('-')[4].split('.')[0]
    except:
        # File couldn't be parsed
        return None


def session_id_from_file_name(file_name):
    """
    Gets the session_id from a *tseries file name

    Parameters
    ----------
    file_name : str
        Example: UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss-CABNP.ntseries.csv.gz

    Returns
    -------
    session_id: str
        Example: 'UM0121_baseline'
    """
    try:
        assert len(file_name.split('-')) > 1
        session_id = file_name.split('-')[0]
        return session_id
    except:
        # File couldn't be parsed
        return None


def scan_type_from_file_name(file_name):
    """
    Identifies scan type from a file name

    Parameters
    ----------
    file_name : str
        Example: UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss-CABNP.ntseries.csv.gz

    Returns
    -------
    scan_type: str
        Example: 'rest_run_2'
    """
    try:
        scan_type = file_name.split('-')[1]
        return scan_type
    except:
        # File couldn't be parsed
        return None


def tseries_unit_from_file_name(file_name):
    """
    Identifies whether a file uses voxels/grey-ordinantes
    (i.e., "dense", "d"), parcels "p", or networks "n"
    from the file's name

    Parameters
    ----------
    file_name : str
        Example: UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss-CABNP.ntseries.csv.gz

    Returns
    -------
    unit: str
        Example: 'n'
    """
    try:
        unit = file_name.split('.')[1][:1]
        assert unit in ['d','p','n']
        return unit
    except:
        # File couldn't be parsed
        return None


def load(file):
    """
    Load neuroimaging data from file. This function loads data from:
        - neuroimaging files
        - plain text files

    Parameters
    ----------
    file : string-like
        path to neuroimaging file

    Returns
    -------
    np.ndarray

    Raises
    ------
    FileExistsError : file does not exist
    RuntimeError : file is empty
    ValueError : file type cannot be determined or is not implemented
    TypeError : input is not string-like

    """
    if not _is_string_like(file):
        raise TypeError(f"expected string-like, got {type(file)}")

    if not Path(file).exists():
        raise FileExistsError(f"file does not exist: {file}")

    if Path(file).stat().st_size == 0:
        raise RuntimeError(f"file is empty: {file}")

    if Path(file).suffix == ".txt":  # text file
        return np.loadtxt(file).squeeze()

    try:
        return _load_neuroim_file(file)
    except TypeError:
        raise ValueError(
            f"expected txt or nii or gii file, got {Path(file).suffix}")


def _is_string_like(obj):
    """ Check whether `obj` behaves like a string. """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def _load_neuroim_file(file):
    """
    Load data contained in a CIFTI2-/GIFTI-format neuroimaging file.

    Parameters
    ----------
    file : filename
        Path to neuroimaging file

    Returns
    -------
    np.ndarray
        Data stored in `file`

    Raises
    ------
    TypeError : `file` has unknown filetype

    """
    try:
        return _load_gifti(file)
    except AttributeError:
        try:
            return _load_cifti2(file)
        except AttributeError:
            raise TypeError(f"file cannot be loaded: {file}")


def _load_gifti(file):
    """
    Load data stored in a GIFTI (.gii) neuroimaging file.

    Parameters
    ----------
    file : filename
        Path to GIFTI-format (.gii) neuroimaging file

    Returns
    -------
    np.ndarray
        Neuroimaging data in `file`

    """
    return nib.load(file).darrays[0].data


def _load_cifti2(file):
    """
    Load data stored in a CIFTI-2 format neuroimaging file (e.g., .dscalar.nii
    and .dlabel.nii files).

    Parameters
    ----------
    file : filename
        Path to CIFTI-2 format (.nii) file

    Returns
    -------
    np.ndarray
        Neuroimaging data in `file`

    Notes
    -----
    CIFTI-2 files follow the NIFTI-2 file format. CIFTI-2 files may contain
    surface-based and/or volumetric data.

    """
    return np.asanyarray(nib.load(file).dataobj).squeeze()


def cifti_parcellate(dense_in, cifti_atlas, parcel_out, logger):
    """
    Use HCP workbench to parcellate a dense cifti file

    Parameters
    ----------
    dscalar_in: str
        Full path to the dense CIFTI file (e.g. *.dtseries.nii, *.dscalar.nii)
    cifti_atlas: str
        Full path to the volume/surface CIFTI dlabel file
    parcel_out: str
        Full path to the parcellated 
    """
    #print('Parcellated Output: {}'.format(parcel_out))
    subprocess.call(['wb_command', '-cifti-parcellate', dense_in, cifti_atlas, 'COLUMN', parcel_out],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
    if logger is not None: 
        logger.info('---')
        logger.info(f'input cifti: {dense_in}')
        logger.info(f'cifti atlas: {cifti_atlas}')
        logger.info(f'output cifti: {parcel_out}')
        logger.info('---')


def get_qunex_dirs(sessions_dir, regex_session_filter):
    '''
    Find QuNex subdirectories that match a certain regex_filter

    Parameters
    ----------
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
    
    Returns 
    ----------
    List[pathlib.Path]
        List of QuNex session directories
    '''
    session_dirs = list()
    for path in Path(sessions_dir).iterdir():
        if regex_session_filter:
            if re.match(regex_session_filter, path.stem):
                session_dirs.append(path)
        elif '_' in path.stem:
            session_dirs.append(path)
    print(f'Qunex Directories Found: {len(session_dirs)}')
    return session_dirs


def load_motion_scrub(session_path, file_prefix):
    """
    Load binary motion-scrub values for a scan.

    Parameters
    ----------
    session_path: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline
    file_prefix: str
        The first part of the BOLD file
        Example: bold1

    Returns
    -------
    use: numpy.array
        boolean values indicating whether corresponding frames should be dropped

    """
    motion_path = session_path.joinpath('images', 'functional', 'movement')
    return np.loadtxt(motion_path.joinpath(file_prefix + '.use'), dtype=int)


def load_parcellation_mapping(parcellation_method):
    """
    Loads the frequently used parcel network mapping data
    for a given parcellation.

    Parameters
    ----------
    parcellation_method: str
        The parcellation name, e.g., 'CABNP'

    Returns
    -------
    mapping: pandas.DataFrame
        A Pandas DataFrame that lists parcels' name,
        parcel index, network, and other details
    """
    assert parcellation_method in reference.parcellation.options
    path = Path(__file__).parent.parent.joinpath('reference/parcellation/mapping/CABNP.csv')
    mapping = pd.read_csv(path)
    return mapping


def load_expected_parcel_order(parcellation_method):
    """
    Get expected order of parcel names as a simple list

    Parameters
    ----------
    parcellation_method: str
        The parcellation name, e.g., 'CABNP'

    Returns
    -------
    ordered_parcel_names: List[str]
        Ordered parcel names
    """
    return list(load_parcellation_mapping(parcellation_method)['ParcelName'].values)


def scrub_motion(session_path, file_prefix, scan_array):
    """
    Removes motion from scan's timeseries

    Parameters
    ----------
    session_path: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline
    file_prefix: str
        The first part of the BOLD file
        Example: bold1
    scan_array: numpy.array
        The raw time series data

    Returns
    -------
    scan_array.T, frames_to_keep: numpy.array, list[int]
        The scrubbed time series along with the list of kept frame
        indices, since these wouldn't otherwise be preserved in
        the returned numpy array. A user can also look at the
        kept frame indices to determine if scrubbing occurred
    """
    use = load_motion_scrub(session_path=session_path, file_prefix=file_prefix)
    frames_to_keep = np.where(use)[0]
    scrubbed_array = scan_array[frames_to_keep, :]
    return scrubbed_array, frames_to_keep


def parse_session_hcp(file_path, scan_types):
    """
    Finds the BOLD ID associated with each named file from
    the session_hcp.txt file of a given session

    Parameters
    ----------
    file_path: str
        Path to the sessions_hcp.txt file
        Example:
            ~/fsx-mount/embarc-20201122-LHzJPHi4/sessions/CU0011_baseline/session_hcp.txt
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
            'bold reward' ]

    Returns
    -------
    mapping: dict:
        The names and their associated BOLD IDs
        Example: {
            'bold ert': '1', 'bold rest run-1': '2',
            'bold rest run-2': '3', 'bold reward': '4'
        }
    """

    # Build the mapping from rest/task to the bold scan, e.g., bold1, bold2, etc.
    with open(file_path, 'r') as f:
        info = f.read()
    mapping = dict()

    # The regex | means "or", i.e., allowed to match any of the scan_types
    scan_type_regex = '|'.join(scan_types)

    # Find all of the matches
    matches = re.findall(f'bold([0-9]+).+:({scan_type_regex})\n', info)
    for bold_id, bold_name in matches:
        mapping[bold_name] = bold_id

    # Try the first known HCP format if no matches found after standard mapping
    if len(mapping) == 0:
        matches = re.findall(f'bold([0-9]+):.+\s:\s({scan_type_regex}).+\n', info)
        for bold_id, bold_name in matches:
            mapping[bold_name] = bold_id

    return mapping


def stack_pconn_matrices(data_dir, feature, glob_string, session_dir):
    """
    Glob a set of pconn.nii CIFTI files, read, stack, stack, and return 

    Parameters
    -----------
    data_dir: str|Path
        Directory with the pconn.nii files to glob
    glob_string: str
        Suffix of the pconn.nii files to glob
            e.g. "_atlas-CABNP_stat-pearsonRtoZ.pconn.nii"
    session_dir: Path
        Session directory Path. This is just used for the session_id value

    Returns
    ----------
    dat_stack: np.matrix
        Stacked 2D numpy connectivity matrices.
        [NxNxi]
    meta_df: pd.DataFrame
        Corresponding metadata for stacked numpy
        connectivity matrices
        [col x i]
    parcel_list: list
        List of parcel names corresponding to the rows/columns
        of connectivity matrices
    """
    # path to the intermediate "refined" data elements
    feature_dir = Path(data_dir, session_dir.stem, 'functional', feature)

    # compile individual feature files 
    read_files = [x for x in feature_dir.rglob(f'*{glob_string}')]
    
    meta_list = []
    df_list   = []
    for read_file in read_files:
        try: 
            fname       = str(read_file).split('/')[-1]
            file_parts  = extract_filename_components(fname)
            meta_data   = pd.DataFrame(file_parts, index=[0])
            meta_data.insert(1, 'feature', feature)

            if 'pconn.nii' in str(read_file).split('/')[-1]:
                pconn_cii  = nb.load(read_file)
                dat_matrix = pconn_cii.dataobj
                df_list.append(np.array(dat_matrix))
                meta_list.append(meta_data)
                parcel_list = [p.name for p in pconn_cii.header.matrix[0].parcels]
        except Exception as e: 
            print(e)
            pass

    if len(df_list) > 0:
        dat_stack = np.stack(df_list)
        meta_df = pd.concat(meta_list)
        return dat_stack, meta_df, parcel_list


def copy_dtseries_files(study_name, session_id, scan_num, scan_name, scan_path, out_dir, logger):
    """
    Copy dtseries.nii file from QuNex directory into
    the raw/functional directory. Also format the
    filename to a BIDS-ish descriptive style.

    Parameters
    -----------
    study_name: str
        A descriptive name of the imaging project/QuNex output being processed
        Example: EMBARC
    session_id: str
        Session ID in the QuNex directory
        Example: TX0038_wk1
    scan_num:
        Scan number, as noted in the QuNex "hcp_session.txt" file
    scan_name:
        Scan name, as noted in the QuNex "hcp_session.txt" file
    scan_path:
        Full path to the dtseries file in QuNex sessions dir
    out_dir:
        Where to copy the dtseries.nii file
    logger
    
    Returns
    -----------
    str
        Full path to newly copied/formatted dtseries.nii file
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    # clean the scan name to be shorter/more readable
    bold_name_clean = str(scan_name).replace('bold', '').replace(' ','').replace('-','_')
    
    # create copy cifti filename
    fname = str(scan_path).split('/')[-1]

    # create new destination filename    
    file_descriptor = fname.replace(f'bold{scan_num}_', '').split('.')[0]
    new_fname       = f'sub-{session_id}_task-{bold_name_clean}_proc-{file_descriptor}_study-{study_name}.dtseries.nii'
    dest_cii        = Path(out_dir, new_fname)

    # do the copy
    if not dest_cii.exists():
        os.symlink(scan_path, dest_cii)
    #shutil.copy(scan_path, dest_cii)

    if logger is not None: 
        logger.info(f'FUNCTION: {inspect.stack()[0][3]}')
        logger.info(f'SOURCE: {scan_path}')
        logger.info(f'DEST:   {dest_cii}')
        logger.info('\n')
    return dest_cii


def get_dtseries_paths(scans, input_scan_pattern, session_dir):
    """
    Retrieve full paths to the dtseries.nii file to be processed.

    Parameters
    -----------
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
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions

    Returns
    -----------
    List[tuples]
        List of tuples with dtseries.nii information
        (scan_number, scan_name, scan_path)
    """
    # read scan number to name mapping
    scan_dict = parse_session_hcp(Path(session_dir, 'session_hcp.txt'), scans)
    scan_dict_rev = {val:key for key,val in scan_dict.items()}

    # make a dict linking scan numbers to absolute cifti paths
    orig_scan_dict = get_qunex_ciftis(session_dir, input_scan_pattern, scan_dict)

    scan_list = []
    for scan_num in orig_scan_dict.keys():
        scan_list.append((scan_num, scan_dict_rev[scan_num], orig_scan_dict[scan_num]))
    return scan_list


def motion_censor(session_dir, scan_num, scan_path, logger):
    """
    Remove high-motion frames from a CIFTI file

    Parameters
    ----------
    session_dir: str
        Full path to the QuNex directory for the session being processed. 
    scan_num: 
        Scan number, as noted in the QuNex "hcp_session.txt" file
    scan_path: 
        Full path to the dtseries file in QuNex sessions dir
    """
    # read motion scrub information
    censor_frames = load_motion_scrub(session_dir, f'bold{scan_num}')
    num_censor_frames = np.sum([censor_frames == 0])

    # create censored cifti filename
    cii_splits = str(scan_path).split('_study-')
    censor_scan_path = f'{cii_splits[0]}_frameCensor_study-{cii_splits[1]}'

    # actually censor the dtseries file
    censor_dtseries(cii_in=scan_path, censor_frames=censor_frames, cii_out=censor_scan_path)
    if logger is not None: 
        logger.info(f'FUNCTION: {inspect.stack()[0][3]}')
        logger.info(f'input cifti:     {scan_path}')
        logger.info(f'total frames:    {len(censor_frames)}')
        logger.info(f'censored frames: {num_censor_frames}')
        logger.info(f'output cifti:    {censor_scan_path}')
        logger.info('\n')

    return censor_scan_path

def concatenate_ciftis(cii_list, concatenate_scan_name, out_dir, logger ):
    """
    Concatenate a list of CIFTIs (in time)

    Parameters
    ----------
    cii_list: List[str]
        List of ciftis to be combined. 
    concatenate_scan_name: str
        Descriptive name for the newly created 
        concatenated CIFTI.
        Example: rest_concatenated
    out_dir: str
        Where to save the new concatenated file
    
    Returns
    ----------
    str
        Full path to the newly created dtseries.nii file
    """           
    # create concatenated CIFTI file
    fname = cii_list[0].split('/')[-1]
    file_parts  = extract_filename_components(fname)

    sub   = file_parts['sub']
    proc  = file_parts['proc']
    study = file_parts['study']

    concat_fname = f'sub-{sub}_task-{concatenate_scan_name}_proc-{proc}_study-{study}.dtseries.nii'
    concat_cii   = Path(out_dir, concat_fname)

    concat_cmd = [WB_COMMAND, '-cifti-merge', concat_cii]
    for cii_file in cii_list:
        concat_cmd += ['-cifti', cii_file]
    subprocess.call(concat_cmd)

    if logger is not None: 
        logger.info(f'FUNCTION: {inspect.stack()[0][3]}')
        for cii in cii_list:
            logger.info(f'input cifti:     {cii}')
        logger.info(f'concat output cifti: {concat_cii}')
        logger.info('\n')

    return concat_cii


def summarize_feat_by_network(csv_file, feat_dim='2D'):
    """
    Average parcel-level feature (e.g. resting-state correlation) by network.

    Parameters
    -----------
    csv_file: str
        Full path to the parcel-level feature data.
        For 2D features, we assume a long format, i.e.
              roi_1  |   roi_2  |  value
            -------------------------------
            parcel_1 | parcel_2 |  0.1234

        For 1D features, we assume a wide format, i.e.
            session_id | src_file    | parcel_1 | parcel_2
            ------------------------------------------------
            TX0037_wk1 | f.pconn.nii | 0.1234   | 0.5678

    feat_dim: str
        Can be '2D' or '1D'
    """
    def _infer_atlas_type(parcel_list):
        if any(parcel_list.str.contains('Orbito-Affective')):
            return 'CABNP'
        elif any(parcel_list.str.contains('17Networks_LH_VisCent_ExStr_1')):
            return 'YeoPlus'
        else:
            return 'something'

    # read feature csv file
    feat_df = pd.read_csv(csv_file, compression='gzip')

    # get a list of parcel names
    if feat_dim == '2D':
        parcel_list = feat_df['roi_1']
    else: 
        parcel_list = pd.Series([f for f in feat_df.columns if f not in ['session_id', 'src_file']])
    
    # use atlas names to figure out which atlas we're analyzing
    # currently this is just "YeoPlus" and "CABNP"
    atlas_type = _infer_atlas_type(parcel_list)

    # get the network-wise averages
    if feat_dim == '2D':
        long_df, wide_df = avg_2d_features(feat_df, atlas_type, csv_file)
    elif feat_dim == '1D':
        wide_df = avg_1d_features(feat_df, atlas_type, csv_file)


def avg_1d_features(feat_df, atlas_type, csv_file):
    """
    Average a feature by network

    Parameters
    -----------
    feat_df: pd.DataFrame
        A wide dataframe with parcels as columns, e.g.
            session_id | src_file    | parcel_1 | parcel_2
            ------------------------------------------------
            TX0037_wk1 | f.pconn.nii | 0.1234   | 0.5678
    atlas_type: str
        CABNP, YeoPlus
    csv_file: str
        Full path to the csv file containing the 'feat_df' data
    
    Returns:
    pd.DataFrame
        Dataframe with the network-averaged features. 
    """
    parcel_list = pd.Series([f for f in feat_df.columns if f not in ['session_id', 'src_file']])
    net, reg    = parcel_to_network(parcel_list=parcel_list, atlas_type=atlas_type)
    uniq_nets   = list(set(net))
    avg_feat_dict = {}
    avg_feat_dict['session_id'] = feat_df['session_id']
    avg_feat_dict['src_file']   = feat_df['src_file']
    for uniq_net in uniq_nets:
        net_idxs = np.where(feat_df.columns.str.contains(uniq_net))[0]
        avg_val  = np.mean(feat_df.iloc[:,net_idxs].values)
        avg_feat_dict[uniq_net] = avg_val
    avg_feat_dict = pd.DataFrame(avg_feat_dict)

    # create output filenames  
    file_base = csv_file.stem.split('_stat-')[0]
    fdict     = extract_filename_components(csv_file.stem)
    stat_base = fdict['stat'].split('.csv')[0].split('_long')[0].split('_wide')[0]
    # save wide format to disk
    wide_out  = Path(csv_file.parent, f'{file_base}_stat-{stat_base}_avg_by_network_wide.csv.gz')
    avg_feat_dict.to_csv(wide_out, compression='gzip', index=None)
    return avg_feat_dict


def avg_2d_features(feat_df, atlas_type, csv_file):
    """
    Average a feature by network

    Parameters
    -----------
    feat_df: pd.DataFrame
        A long dataframe with a format like this: 
            roi_1  |   roi_2  |  value
            -------------------------------
            parcel_1 | parcel_2 |  0.1234
    atlas_type: str
        CABNP, YeoPlus
    csv_file: str
        Full path to the csv file containing the 'feat_df' data
    
    Returns:
    pd.DataFrame
        Long version of the dataframe with the network-averaged features. 
    pd.DataFrame
        Wide version of the dataframe with the network-averaged features. 
    """
    # TODO: break this function into sub-functions for easier reading/clarity

    # extract network and region information from the parcel names
    feat_df.loc[:,'roi_1_net'], feat_df.loc[:,'roi_1_reg'] = parcel_to_network(parcel_list=feat_df['roi_1'], atlas_type=atlas_type)
    feat_df.loc[:,'roi_2_net'], feat_df.loc[:,'roi_2_reg'] = parcel_to_network(parcel_list=feat_df['roi_2'], atlas_type=atlas_type)
    # get rid of matrix diagonal
    feat_df = feat_df.loc[feat_df['roi_1'] != feat_df['roi_2']]

    # create combined net/region column 
    if atlas_type == 'YeoPlus':
        feat_df.loc[:,'roi_1_netreg'] = feat_df['roi_1_net'] + '_' + feat_df['roi_1_reg']
        feat_df.loc[:,'roi_2_netreg'] = feat_df['roi_2_net'] + '_' + feat_df['roi_2_reg']
        col_1 = 'roi_1_netreg'
        col_2 = 'roi_2_netreg'
    else: 
        col_1 = 'roi_1_net'
        col_2 = 'roi_2_net'

    # average the feature by network  
    uniq_nets      = list(set(np.concatenate((feat_df[col_1], feat_df[col_2]))))
    network_combos = [combo for combo in itertools.combinations(uniq_nets, 2)]
    avg_feat_list  = []
    avg_wide_list  = {}
    for net1, net2 in network_combos:
        # get the average network-to-network correlation
        vals_1   = feat_df.loc[(feat_df[col_1] == net1) & (feat_df[col_2] == net2), 'value']
        vals_2   = feat_df.loc[(feat_df[col_1] == net2) & (feat_df[col_2] == net1), 'value']
        all_vals = vals_2.append(vals_1)
        mean_val = np.mean(all_vals)
        avg_feat_list.append({'net_1': net1, 'net_2': net2, 'value': mean_val})
        avg_wide_list[f'{net1}__to__{net2}'] = mean_val
    
    # organize the new features into both long and wide format
    net_feat_long_df = pd.DataFrame(avg_feat_list)
    net_feat_wide_df = pd.DataFrame(pd.Series(avg_wide_list)).transpose()
    
    # create output filenames  
    file_base = csv_file.stem.split('_stat-')[0]
    fdict     = extract_filename_components(csv_file.stem)
    stat_base = fdict['stat'].split('.csv')[0].split('_long')[0].split('_wide')[0]
    # save long format to disk
    long_out  = Path(csv_file.parent, f'{file_base}_stat-{stat_base}_avg_by_network_long.csv.gz')
    net_feat_long_df.to_csv(long_out, compression='gzip', index=None)
    # save wide format to disk
    wide_out  = Path(csv_file.parent, f'{file_base}_stat-{stat_base}_avg_by_network_wide.csv.gz')
    net_feat_wide_df.to_csv(wide_out, compression='gzip', index=None)

    return net_feat_long_df, net_feat_wide_df


def parcel_to_network(parcel_list, atlas_type):
    """
    TODO
    """
    if atlas_type == 'CABNP':
        network_list = [re.split('(-[0-9]{2})', x)[0] for x in parcel_list]
        region_list  = [re.split('-', x)[-1] for x in parcel_list]
        return network_list, region_list 

    elif atlas_type == 'YeoPlus':
        # pre-allocate network assignment list
        network_list = pd.Series([None]*len(parcel_list))
        region_list = pd.Series([None]*len(parcel_list))
        # parse Schaefer atlas cortical parcels
        schaefer_ctx_idxs = np.where(parcel_list.str.contains('7Networks_'))[0]
        schaefer_ctx_nets = [re.split('(.{2}Networks_.{2}_)', x)[-1].split('_')[0] for x in parcel_list[schaefer_ctx_idxs]]
        network_list[schaefer_ctx_idxs] = schaefer_ctx_nets
        region_list[schaefer_ctx_idxs] = 'Cortex'
        # parse Buckner cortical parcels
        cblm_idxs = np.where(parcel_list.str.contains('Cerebellum'))[0]
        cblm_nets = [re.split('Cerebellum_17Net_.{2}_', x)[-1].split('_')[0] for x in parcel_list[cblm_idxs]]
        network_list[cblm_idxs] = cblm_nets
        region_list[cblm_idxs] = 'Cerebellum'
        # everything else (subcortex)
        ctx_cblm_idxs = np.concatenate((schaefer_ctx_idxs, cblm_idxs))
        all_idxs    = np.array(range(len(network_list)))
        subctx_idxs = set(all_idxs) - set(ctx_cblm_idxs)
        subctx_nets = [x.split('-RH')[0].split('-LH')[0].split('_LH')[0].split('_RH')[0] for x in parcel_list[subctx_idxs]]
        network_list[subctx_idxs] = subctx_nets
        region_list[subctx_idxs] = 'Subcortex'
        return network_list, region_list



def graph_theory_metrics(csv_gz, session_id, stat_name='graphTheoryPearsonRtoZ'):
    """
    Produce Graph Theory metrics using R script 

    Parameters
    -----------
    csv_gz: str,Path
        Absolute path to a 2d connectivity matrix zipped csv file
    session_id: str
        Session ID number (e.g. CU0075_baseline)
    """
    # define graph theory output
    out_dir = Path(Path(csv_gz).parent.parent, 'graphTheory')
    out_dir.mkdir(exist_ok=True, parents=True)
    # create output file name
    ofile_base = csv_gz.stem.split("_stat-")[0]
    ofile_name = f'{ofile_base}_stat-{stat_name}.csv.gz'
    # full path to the compressed csv output file
    out_file = Path(out_dir, ofile_name)

    subprocess.call(['Rscript', GRAPH_METRICS_SCRIPT, '-f', csv_gz, '-s', session_id, '-o', out_file],
        stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)



    
def bad_voxels_by_parcel(ribbon_dir, cifti_atlas):
    """
    Estimate the number/percent of "bad" voxels within a
    given CIFTI parcellation. Good/Bad voxels is a QC metric
    specific to the HCP pipeline. This function will read data
    in the ribbon HCP directory (i.e. "*goodvoxels.32k_fs_LR.func.gii")

    Parameters
    ------------
    ribbon_dir: str
        Absolute path to the HCP ribbon directory for a QuNex session
    cifti_atlas: str
        Absolute path to the CIFTI parcellation dlabel.nii file
    
    Returns
    ------------
    return_df: pd.DataFrame
        Pandas Dataframe with parcel-wise estimates of "bad" voxels
        or surface data points. 

    """

    def _dlabel_to_df(dlabel_path):
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

    def _surf_data_from_cifti(data, hdr_axis, surf_name):
        """
        TODO
        """
        #cii_axis = dlabel_cii.header.get_axis(1)
        for name, data_indices, model in hdr_axis.iter_structures():  # Iterates over volumetric and surface structures
            if name == surf_name:                                 # Just looking for a surface
                data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
                vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
                surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data

    # read cifti atlas information
    dlabel_cii  = nb.load(cifti_atlas)
    dlabel_data = dlabel_cii.dataobj[0]
    dlabel_hdr  = dlabel_cii.header.get_axis(1)
    dlabel_df   = _dlabel_to_df(cifti_atlas)

    # append LH/RH/VOL bad voxel data here
    bad_vox_list = []

    for hemi in ['L', 'R']:
        # name of structure within the cifti file
        if hemi == 'L':
            cii_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
        elif hemi == 'R':
            cii_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'

        # Read GIFTI file with surface-based voxel quality
        hemi_goodvox_path = Path(ribbon_dir, f'{hemi}.goodvoxels.32k_fs_LR.func.gii')
        hemi_goodvox      = nb.load(hemi_goodvox_path)
        goodvox_arr       = hemi_goodvox.darrays[0].data
        
        # Read GIFTI file with surface-based voxel quality
        hemi_mean_path = Path(ribbon_dir, f'{hemi}.mean.32k_fs_LR.func.gii')
        hemi_mean      = nb.load(hemi_mean_path)
        mean_arr       = hemi_mean.darrays[0].data

        # cifti labels
        hemi_labels = _surf_data_from_cifti(dlabel_data, dlabel_hdr, cii_structure)
        
        # dataframe with voxel/label/numeric indicator of voxel quality
        good_vox_df = pd.DataFrame({'labels':hemi_labels, 'goodvox':goodvox_arr, 'mean':mean_arr})

        # aggregate info about vertices marked with zero
        num_zero  = good_vox_df.groupby('labels').agg(lambda x: np.sum(x == 0))['mean']
        num_ords  = good_vox_df.groupby('labels').agg(lambda x: len(x == 0))['mean']
        percent_goodvox = good_vox_df.groupby('labels').agg(lambda x: np.sum(x == 0)/len(x == 0))['goodvox']
        percent_zero    = good_vox_df.groupby('labels').agg(lambda x: np.sum(x == 0)/len(x == 0))['mean']

        badvox_df = pd.DataFrame({'num_ordinates': num_ords, 
                                    'num_zero_ordinates':num_zero, 
                                    'percent_bad_ordinates':percent_goodvox,
                                    'percent_zero_ordinates':percent_zero}).reset_index()
        # append to list
        bad_vox_list.append(badvox_df)

    # separate volume data from cifti 
    vol_out = cifti_atlas.replace('.dlabel.nii', '.nii')
    if not Path(vol_out).exists():
        subprocess.call([WB_COMMAND, '-cifti-separate', cifti_atlas, 'COLUMN', '-volume-all', vol_out],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    # Read volumetric label data
    vol_nii       = nb.load(vol_out)
    vol_label_dat = np.array(vol_nii.dataobj)
    vol_labels    = np.unique(vol_label_dat)

    # good/bad voxels in volume space
    vol_goodvox     = nb.load(Path(ribbon_dir, f'goodvoxels.nii.gz'))
    vol_goodvox_dat = np.array(vol_goodvox.dataobj)

    # good/bad voxels in volume space
    vol_mask     = nb.load(Path(ribbon_dir, f'mask.nii.gz'))
    vol_mask_dat = np.array(vol_mask.dataobj)

    vol_list = []
    for label in vol_labels:
        roi_goodvox = vol_goodvox_dat[vol_label_dat == label]
        roi_mask = vol_mask_dat[vol_label_dat == label]

        num_ords = len(roi_goodvox)
        num_zero = np.sum(roi_goodvox == 0)
        num_zero_mask = np.sum(roi_mask == 0)
        percent_goodvox  = num_zero/num_ords
        percent_mask = num_zero_mask/num_ords
        badvox_df = pd.Series({'labels': label, 
                                    'num_ordinates': num_ords,
                                    'num_zero_ordinates':num_zero, 
                                    'percent_bad_ordinates':percent_goodvox, 
                                    'percent_zero_ordinates':percent_mask})
        vol_list.append(badvox_df)
    vol_df = pd.DataFrame(vol_list)
    bad_vox_list.append(vol_df)
    goodvox_df = pd.concat(bad_vox_list)
    goodvox_df = goodvox_df.loc[goodvox_df.labels != 0]

    # merge with ROI label/name df
    return_df = dlabel_df.merge(goodvox_df, right_on='labels', left_on='roi_num')
    return return_df



def cii_falff_wrapper(scan_path, raw_out_dir):
    """
    Wrapper function for calculating
    ALFF, fALFF, and RSFA for a dtseries.nii file

    Parameters
    ----------
    scan_path: str|Path
        Absolute path to a dtseries.nii CIFTI file
    raw_out_dir: str|Path
        Output directory

    Returns
    ----------
    dscalar_list: List
        List of absolute paths for produced
        (f)ALFF/RSFA dscalar.nii files
    """
    # load cifti
    cii_obj       = nb.load(scan_path)
    dtseries_mat  = cii_obj.dataobj
    
    # set (f)ALFF parameters
    min_low_freq   = 0.01
    max_low_freq   = 0.08
    min_total_freq = 0
    max_total_freq = 0.25
    falff_inputs   = [min_low_freq, max_low_freq, min_total_freq, max_total_freq]

    # get falff/alff/rsfa for each cifti grayordinate
    alff_list = []
    for ordinate in range(dtseries_mat.shape[1]):
        alff_out = calculate_amps(dtseries_mat[:,ordinate], *falff_inputs)
        alff_list.append(alff_out)
    
    # create dataframe
    alff_df = pd.DataFrame(alff_list)
    alff_df.columns = ['fALFF', 'ALFF', 'RSFA']

    # create amplitude directory, if needed
    Path(raw_out_dir, 'Amplitude').mkdir(exist_ok=True, parents=True)

    # write dscalar outputs
    dscalar_list = []
    for measure in ['fALFF', 'ALFF', 'RSFA']:
        # copy dtseries header
        cii_hdr  = cii_obj.header.copy()
        # reshape data to proper format
        new_data = np.array(alff_df[measure]).reshape(1,-1)
        grayordinate_axis = cii_hdr.get_axis(1)
        scalar_axis = nb.cifti2.cifti2_axes.ScalarAxis(['default'])
        # save
        scan_fname = str(scan_path).split('/')[-1]
        new_dscalar = scan_fname.replace('.dtseries.nii', f'_stat-{measure}.dscalar.nii')

        dscalar_out = Path(raw_out_dir, 'Amplitude', new_dscalar)
        new_cii = nb.cifti2.Cifti2Image(new_data, header=[scalar_axis, grayordinate_axis])
        nb.save(new_cii, dscalar_out)
        dscalar_list.append(dscalar_out)
    return dscalar_list
    

def calculate_amps(timeseries, min_low_freq, max_low_freq, min_total_freq, max_total_freq):
    """
    Calculate ALFF, fALFF, RSFA for a timeseries

    Parameters
    ----------
    timeseries: np.array
        Timeseries array
    min_low_freq: float
        Minimum low frequency, used for fractional ALFF
    max_low_frew: float
        Maximum low frequency, used for fractional ALFF
    min_total_freq: float
        Minimum total frequency
    max_total_freq: float
        Maximum total frequency

    Returns
    ----------
    alff: float
        Amplitude of Low Frequency Fluctuation
    falff: float
        Fractional Amplitude of Low Frequency Fluctuation
    rsfa: float
        Resting State Functional Amplitude
    """
    n = len(timeseries)
    time = (np.arange(n))*2

    # Takes fast Fourier transform of timeseries
    fft_timeseries = fft(timeseries)
    
    # Calculates frequency scale
    freq_scale = np.fft.fftfreq(n, 1/1)

    # Calculates power of fft
    mag = (abs(fft_timeseries))**0.5

    # Finds low frequency range (0.01-0.08) and total frequency range (0.0-0.25)
    low_ind   = np.where((float(min_low_freq) <= freq_scale) & (freq_scale <= float(max_low_freq)))
    total_ind = np.where((float(min_total_freq) <= freq_scale) & (freq_scale <= float(max_total_freq)))

    # Indexes power to low frequency index, total frequency range
    low_power = mag[low_ind]
    total_power = mag[total_ind]

    # Calculates sum of lower power and total power
    low_pow_sum = np.sum(low_power)
    total_pow_sum = np.sum(total_power)

    # Calculates alff as the sum of amplitudes within the low frequency range
    alff = low_pow_sum

    # Calculates falff as the sum of power in low frequnecy range divided by sum of power in the total frequency range
    falff = np.divide(low_pow_sum, total_pow_sum)

    # simple temporal standard dev of the unfiltered timeseries 
    rsfa = np.std(timeseries)
    return alff, falff, rsfa

def get_qunex_ciftis(session_dir, input_scan_pattern, scan_dict):
    """
    Retrieve the full path to the CIFTI input files within QuNex output directory.

    Parameters
    ----------
    session_dir: str
        Full path to the QuNex directory for the session being processed. 
    """
    cii_dict = {}
    for scan_num in scan_dict.values():
        glob_file = [x for x in Path(session_dir, 'images', 'functional').glob(f'bold{scan_num}*{input_scan_pattern}')]
        if len(glob_file) == 1:
            cii_file = glob_file[0]
            cii_dict[scan_num] = cii_file
    return cii_dict



def compute_gbc(conn_file, stat_name, out_dir):
    """
    Compute Global Brain Connectivity from pconn.nii file

    Parameters
    ----------
    conn_file: str
        Full path to the pconn.nii file to process
        Example: sub-TX0037_wk1_task-restrun_1_proc-Atlas_s_hpss_res-mVWMWB_lpss_frameCensor_study-EMBARC_atlas-CABNP_stat-PearsonR.pconn.nii
    stat_name: str
        Descriptive statistics name
        Example: PearsonR_GBC
    out_dir: str
        Full path to where you want to save the GBC data

    Returns
    ----------
    str
        Full path to the GBC csv file. 
    """

    # Global Brain Connectivity
    # -----
    gbc_dir = Path(out_dir, 'GBC')
    gbc_dir.mkdir(exist_ok=True, parents=True)

    fname_dict = extract_filename_components(conn_file.stem)
    session_id = fname_dict['sub']
    file_base  = conn_file.stem.split('_stat-')[0]

    # compute GBC
    gbc_pscalar = Path(gbc_dir, f'{file_base}_stat-{stat_name}.pscalar.nii')
    subprocess.call([WB_COMMAND, '-cifti-reduce', conn_file, 'MEAN', gbc_pscalar, '-only-numeric'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
    gbc_cii = nb.load(gbc_pscalar)
    gbc_vals = gbc_cii.dataobj[0]
    
    # format into dataframe
    gbc_df = pd.DataFrame(gbc_vals).transpose()
    parcel_list = [p.name for p in gbc_cii.header.matrix[1].parcels]
    gbc_df.columns = parcel_list

    # format to data frame
    gbc_df      = pd.DataFrame(gbc_vals).transpose()
    gbc_df.columns = parcel_list

    # insert session ID and name of the source file
    gbc_df.insert(0, 'session_id', session_id)
    src_file = str(conn_file).split('/')[-1]
    gbc_df.insert(1, 'src_file', src_file)
    
    # save
    gbc_file = Path(gbc_dir, f'{file_base}_stat-{stat_name}.csv.gz')
    gbc_df.to_csv(gbc_file, index=None, compression='gzip')

    return gbc_file


def censor_dtseries(cii_in, censor_frames, cii_out):
    """
    Censor frames in a *.dtseries file

    Parameters
    ----------
    cii_in: str
        Full path to the dtseries (or ptseries) file to censor
    censor_frames: np.array
        Array of 0 (censor) and 1 (keep).
        Must be the same length as the number of timepoints in the series CIFTI file.
    cii_out: str
        Full path to the censored CIFTI file.
    """
    # Load CIFTI file
    cii_obj = nb.load(cii_in)
    # pull data
    cii_dat = cii_obj.get_fdata()
    # pull out header to modify for the new CIFTI
    cii_hdr    = cii_obj.header
    
    # make sure that the length of censor_frames array is the 
    # same length as the CIFTI series file
    num_tr, num_vertices = cii_obj.dataobj.shape
    assert len(censor_frames) == num_tr

    # get rid of censored frames
    censor_dat = cii_dat[censor_frames == 1,]
    
    # create new header
    ax_0       = nb.cifti2.SeriesAxis(start=0, step=cii_hdr.get_index_map(0).series_step, size=censor_dat.shape[0]) 
    ax_1       = cii_hdr.get_axis(1)
    new_h      = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
    censor_cii = nb.cifti2.cifti2.Cifti2Image(dataobj=censor_dat, header=new_h)

    # save to disk
    censor_cii.to_filename(cii_out)



def cifti_correlate(cii_file, out_dir, stat_name, fisher=False, covariance=False):
    """
    Wrapper around HCP workbench to correlate a *series.nii CIFTI file

    Parameters
    ----------
    cii_file: str
        Full path to the input CIFTI file (e.g. *.dtseries.nii, *.ptseries.nii)
    conn_file: str
        Full path to the output CIFTI file (e.g. *.pconn.nii)
    csv_file: str
        Full path to the output csv file containing the correlation matrix
    fisher: bool
        If True, do Fisher r-to-z transform
    covariance: bool
        If True, output covariance instead of correlation
    """
    fname      = cii_file.split('/')[-1]
    fname_dict = extract_filename_components(fname)
    file_base = fname.replace('.ptseries.nii','').replace('.dtseries.nii','')
    
    stat_out_dir = Path(out_dir, stat_name)
    stat_out_dir.mkdir(exist_ok=True, parents=True)

    # file outputs
    conn_file   = Path(stat_out_dir, f'{file_base}_stat-{stat_name}.pconn.nii')
    csv_file    = Path(stat_out_dir, f'{file_base}_stat-{stat_name}.csv.gz')
    csv_long_file = Path(stat_out_dir, f'{file_base}_stat-{stat_name}_long.csv.gz')
    csv_wide_file = Path(stat_out_dir, f'{file_base}_stat-{stat_name}_wide.csv.gz')

    # workbench correlation command
    corr_cmd = [WB_COMMAND, '-cifti-correlation', cii_file, conn_file]
    
    if fisher == True:
        corr_cmd += ['-fisher-z']

    if covariance == True:
        corr_cmd += ['-covariance']
    subprocess.call(corr_cmd)
    
    if csv_file != None:
        # read the parcellated connectivity file
        conn_cii    = nb.load(conn_file)
        # extract names of each parcel
        parcel_list = [p.name for p in conn_cii.header.matrix[0].parcels]

        # create pandas dataframe with the connectivity matrix (label cols/rows)
        conn_matrix         = pd.DataFrame(np.matrix(conn_cii.dataobj))
        conn_matrix.columns = parcel_list
        conn_matrix.index   = parcel_list
        
        # save to disk
        conn_matrix.to_csv(csv_file, index=None, compression='gzip')

    melt_corr_matrix(pconn_file=conn_file, session_id=fname_dict['sub'], out_file=csv_long_file)
    flatten_corr_matrix(pconn_file=conn_file, session_id=fname_dict['sub'], out_file=csv_wide_file)

    return conn_file, csv_file, csv_long_file, csv_wide_file



def melt_corr_matrix(pconn_file, session_id, out_file):
    """
    Read a cifti pconn.nii file and write a flat csv.
    The first two columns of the csv will be `session_id` and `src_file`
    Each column takes the form of `{roi_1}__to__{roi_2}`

    Paramaters
    ----------
    pconn_file: str
        Full path to a CIFTI parcellated connectivity pconn.nii file
    session_id: str
        ID for the current session
        e.g. 'TX0033_baseline'
    out_file: str
        Full path to the flat connectivity csv file
    
    Notes
    ----------
    Parcel labels are take directly from the CIFTI metadata.
    """
    # read the parcellated connectivity file
    conn_cii    = nb.load(pconn_file)
    # extract names of each parcel
    parcel_list = [p.name for p in conn_cii.header.matrix[0].parcels]

    # create pandas dataframe with the connectivity matrix (label cols/rows)
    conn_matrix         = pd.DataFrame(np.matrix(conn_cii.dataobj))
    conn_matrix.columns = parcel_list
    conn_matrix.index   = parcel_list
    conn_matrix_reset   = conn_matrix.reset_index()

    # melt the connectivity matrix (both upper and lower triangle)
    conn_upper = conn_matrix.where(np.triu(np.ones(conn_matrix.shape)).astype(bool))
    conn_upper_reset_melt   = conn_upper.stack().reset_index()
    conn_upper_reset_melt.columns = ['roi_1', 'roi_2', 'value']
    conn_upper_reset_melt.to_csv(out_file, index=None, compression='gzip')



def flatten_corr_matrix(pconn_file, session_id, out_file):
    """
    Read a cifti pconn.nii file and write a flat csv.
    The first two columns of the csv will be `session_id` and `src_file`
    Each column takes the form of `{roi_1}__to__{roi_2}`

    Paramaters
    ----------
    pconn_file: str
        Full path to a CIFTI parcellated connectivity pconn.nii file
    session_id: str
        ID for the current session
        e.g. 'TX0033_baseline'
    out_file: str
        Full path to the flat connectivity csv file
    
    Notes
    ----------
    Parcel labels are take directly from the CIFTI metadata.
    """
    # read the parcellated connectivity file
    conn_cii    = nb.load(pconn_file)
    # extract names of each parcel
    parcel_list = [p.name for p in conn_cii.header.matrix[0].parcels]

    # create pandas dataframe with the connectivity matrix (label cols/rows)
    conn_matrix         = pd.DataFrame(np.matrix(conn_cii.dataobj))
    conn_matrix.columns = parcel_list
    conn_matrix.index   = parcel_list
    conn_matrix_reset   = conn_matrix.reset_index()

    # melt the connectivity matrix (both upper and lower triangle)
    conn_upper  = conn_matrix.where(np.triu(np.ones(conn_matrix.shape)).astype(bool))
    conn_upper_reset_melt   = conn_upper.stack().reset_index()
    conn_upper_reset_melt.columns = ['roi_1', 'roi_2', 'value']

    #conn_melted = conn_matrix_reset.melt(id_vars='index')
    conn_wide   = pd.DataFrame(conn_upper_reset_melt.value).transpose()
    conn_wide.columns = conn_upper_reset_melt['roi_1'] + '__to__' + conn_upper_reset_melt['roi_2']

    # add session id and the pconn.nii filname
    conn_wide.insert(0, 'session_id', session_id)
    src_file = str(pconn_file).split('/')[-1]
    conn_wide.insert(1, 'src_file', src_file)

    # save to disk
    conn_wide.to_csv(out_file, index=None, compression='gzip')



