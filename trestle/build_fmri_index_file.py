import ast
import pathlib
from pathlib import Path
import re
from typing import List

import boto3
import click
import numpy as np
import pandas as pd

from utils.neuroimage import parse_session_hcp


def regex_extract_time_point(pattern, session_id):
    """
    Gets the time point from a session ID. Returns
    None if the session ID could not be found or
    if there is an error parsing the session ID

    Parameters
    ----------
    pattern : str
        The regex pattern used to get the time point
        from the sessiond ID. For example, to get the
        embarc time point from 'TX0038_baseline' or
        'CU0009_wk1' use '.{6}_(baseline|wk1|wk2)'
    session_id: str
        Example: 'TX0038_baseline', 'CU0009_wk1'

    Returns
    -------
    time_point: str
        The time point as a string or None if there
        was an error in the regex search or if the
        time point could not be found
    """

    try:
        result = re.match(pattern, session_id)
        # group(0) is entire match, group(1) is first group
        return result.group(1)
    except:
        return None


def search_s3_files(bucket, prefix, regex):
    """
    Gets all of the S3 file paths matching a regex
    pattern. Note that S3 does support a more efficient
    operation. You should help out the search by
    specifying a 'prefix' (official S3 term) to narrow
    the breadth of the search. Otherwise this function
    will have to loop through all files in the bucket
    which could take several minutes or more.

    Parameters
    ----------
    bucket : str
        The bucket to search. Example: 'neuro-trestle-ds'
    prefix: str
        AWS S3 official terminology for "S3 path" is
        "prefix." The function will search all files under
        this S3 path. Example:
            'trestle/data_sources/hcp/fmri/processed_data/tseries'
    regex: str
        Regex pattern used to filter S3 files. Example:
            '.*([0-9]{6})-(rest|task)-Atlas_s_hpss_res-mVWMWB_lpss.dtseries.csv.gz'

    Returns
    -------
    search_results: list[str]
        A list of all the matching paths. Note that the values
        are not returned with a starting '/'. Example: [
            'trestle/data_sources/fake_study/fmri/processed_data/tseries/123-rest-Atlas_s_hpss_res-mVWMWB_lpss.dtseries.csv.gz',
            'trestle/data_sources/fake_study/fmri/processed_data/tseries/456-rest-Atlas_s_hpss_res-mVWMWB_lpss.dtseries.csv.gz',
            'trestle/data_sources/fake_study/fmri/processed_data/tseries/123-task-Atlas_s_hpss_res-mVWMWB_lpss.dtseries.csv.gz',
        ]
    """
    if prefix[-1] != '/':
        prefix += '/'
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    search_results = []
    for response in response_iterator:
        if 'Contents' not in response:
            print('No search results found!')
            return search_results
        for object_data in response['Contents']:
            if re.search(regex, object_data['Key']):
                search_results.append(object_data['Key'])
    print(f'Found {len(search_results)} dtseries files in S3')
    return search_results


@click.command()
@click.option('--dtseries-s3-bucket', type=str)
@click.option('--dtseries-s3-path', type=str)
@click.option('--dtseries-regex', type=str)
@click.option('--output-file', type=click.Path())
@click.option('--trestle-s3-bucket', type=str)
@click.option('--trestle-s3-path', type=str)
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--regex-session-filter')
@click.option('--scan-types', multiple=True)
@click.option('--time-point-regex', type=str, default=None)
@click.option('--time-point-mapping-str', type=str, default=None)
@click.option('--time-point-unit', type=str, default=None)
def main(
        dtseries_s3_bucket: str,
        dtseries_s3_path: str,
        dtseries_regex: str,
        output_file: pathlib.PosixPath,
        trestle_s3_bucket: str,
        trestle_s3_path: str,
        sessions_dir: pathlib.PosixPath,
        regex_session_filter: str,
        scan_types: List[str],
        time_point_regex: str,
        time_point_mapping_str: str,
        time_point_unit: str
):
    """
    Creates a study's Trestle index file and uploads it to S3

    Parameters
    ----------
    dtseries_s3_bucket: str
        The bucket to search for dtseries flattened text files.
        Example: 'neuro-trestle-ds'
    dtseries_s3_path: str
        AWS S3 official terminology for "S3 path" is "prefix."
        The function will search all dtseries files under this
        S3 path. Example:
            'trestle/data_sources/hcp/fmri/processed_data/tseries'
    dtseries_regex: str
        Regex pattern used to filter S3 files. Example:
            '.*([0-9]{6})-(rest|task)-Atlas_s_hpss_res-mVWMWB_lpss.dtseries.csv.gz'
    output_file: pathlib.PosixPath
        Local path to save the index file. The index file
        is saved to S3, so output_file can be a temp location
        Example: '~/hcp.trestle_project.fmri.processed_data.tseries.MD'
    trestle_s3_bucket: str
        The destination bucket for the Trestle index file.
        Example: 'neuro-trestle-ds'
    trestle_s3_path: str
        Destination S3 prefix (path) for the Trestle index
        file. Example: 'trestle/data_sources/hcp/metadata/fmri/processed_data'
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
        Example: '(session_a|session_b|session_c)' to capture only
            'session_a', 'session_b', and 'session_c' sessions
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
    time_point_regex: str
        Optional CLI regex arg to extract a time point from
        a session ID. Many studies do not have a time point.
        The regex pattern should have exactly one group that
        you specify in (). Note that | signifies 'or' in regex
        Example: '.{6}_(baseline|wk1|wk2)'
    time_point_mapping_str: str
        Optional CLI arg that will be used to convert the
        output of the regex match to a numeric value. The
        code converts your mapping string to a dictionary.
        This field is required if you use time_point_regex.
        Example: --time-point-mapping-str "{'baseline':0, 'wk1':1, 'wk2':2}"
    time_point_unit: str
        Optional CLI arg to specify time point units. For
        embarc this would simply be 'week' though for
        other studies it could be day, month, hour, etc.
    """

    files = search_s3_files(
        bucket=dtseries_s3_bucket,
        prefix=dtseries_s3_path,
        regex=dtseries_regex
    )

    # ignore the first folder in the Trestle S3 bucket, which is 'trestle'
    filepaths = ['/'.join(f.split('/')[1:]) for f in files]

    session_ids = np.array([f.split('-')[0].split('/')[-1] for f in files]).T
    scans = np.array([f.split('-')[1] for f in files]).T
    df = pd.DataFrame(
        data={
            'filepath': filepaths,
            'session_id': session_ids,
            'scan': scans,
            'bold_id': np.nan
        }
    )
    df.set_index(keys=['session_id', 'scan'], inplace=True)

    # Get bold_id
    session_dirs = list()
    for path in Path(sessions_dir).iterdir():
        if re.match(regex_session_filter, path.name):
            session_hcp_path = path.joinpath('session_hcp.txt')
            mapping = parse_session_hcp(
                file_path=str(session_hcp_path), scan_types=scan_types)
            session = path.name

            # add scan number info
            for scan, bold in mapping.items():
                # update scan type, eg bold rest run-1 -> rest_run_1 etc
                key = scan.replace('bold ', '').replace(' ', '_').replace('-', '_')
                # update df
                df.loc[(session, key), 'bold_id'] = bold

    # drop where mapping exists but file doesnt
    df = df[df['filepath'].notnull()]

    # reset index and sort by session_id
    df = df.reset_index().sort_values(by='session_id', axis=0)

    # add time point if available for the study
    if time_point_mapping_str and time_point_regex and time_point_unit:

        # Convert dict string to an actual dict
        # https://stackoverflow.com/a/988251/554481
        timepoint_mapping = ast.literal_eval(time_point_mapping_str)

        time_point = df.session_id.apply(lambda x: regex_extract_time_point(time_point_regex, x))
        df['time_point'] = time_point.map(timepoint_mapping)
        df['time_point_unit'] = time_point_unit

    # add rest binary column
    df['rest'] = df['scan'].str.contains(pat='rest', case=False)

    # add resolution column (starting point: voxels)
    df['resolution'] = 'voxel'

    # write to file
    df.to_csv(output_file, index=False, sep=',')

    # Use the AWS resource's IAM role or laptop's default aws profile
    s3 = boto3.resource('s3')

    # Remove trailing and leading slashes so S3 doesn't complain
    trestle_s3_path = trestle_s3_path.strip('/')
    output_file_name = Path(output_file).name
    s3.Bucket(trestle_s3_bucket).upload_file(
        f'{output_file}',
        f'{trestle_s3_path}/{output_file_name}')


if __name__ == '__main__':
    main()
