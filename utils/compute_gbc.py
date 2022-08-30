import pathlib
from multiprocessing import Pool
from pathlib import Path

import click
import pandas as pd
import tqdm

import features.gbc
import reference.parcellation
from utils.pairwise import pairwise_pearson
from utils.neuroimage import *


def make_session_row(input_path: pathlib.PosixPath):
    """
    Calculates GBC values for a given session's tseries file.
    This function produces a single row in what eventually becomes
    a consolidated Pandas DataFrame of all sessions' GBC values
    after the process pool finishes

    Parameters
    ----------
    input_path : pathlib.PosixPath
    Where to find the tseries input file. Example:
        '/trestle/data_sources/embarc/fmri/transient/tseries/UM0121_baseline-concatenated_rest-Atlas_s_hpss_res-mVWMWB_lpss-CABNP.ptseries.csv.gz'

    Returns
    -------
    row: dict
        A dictionary that has a key-value pair for each
        parcellation and its GBC value. Also contains the
        session ID so that the row can be uniquely identified
        in the Pandas DataFrame
    """

    # Get parcellation method
    parcellation_method = parcellation_from_file_name(input_path.name)

    # Get parcellation names
    parcellation_mapping = load_parcellation_mapping(parcellation_method)
    parcel_names_expected_order = parcellation_mapping['ParcelName'].values

    # Read time series from disk
    time_series = pd.read_csv(input_path)

    # Check that the order and spelling of parcel names match
    if not valid_parcel_schema(parcel_names_expected_order, time_series.columns):
        raise Exception(f'{input_path} has an invalid parcel schema!')

    # Ignore motion-scrubbed frames
    time_series.drop(time_series[time_series.use == 0].index, inplace=True)

    # Compute correlation matrix
    correlation_matrix = pairwise_pearson(
        time_series.loc[:, parcel_names_expected_order].to_numpy().T)

    # Compute GBC values
    gbc_values = features.gbc.compute(correlation_matrix)

    # Associate each label with its value
    row = dict(zip(parcel_names_expected_order, gbc_values))

    # Include session ID to differentiate row when later put into Pandas
    file_name = input_path.stem
    session_id = session_id_from_file_name(file_name)
    row['session_id'] = session_id
    return row


@click.command()
@click.option('--tseries-unit', type=click.Choice(['d','p','n']))
@click.option('--parcellation-method', required=False,
    type=click.Choice(reference.parcellation.options))
@click.option('--input-folder', type=click.Path(exists=True))
@click.option('--input-file-pattern')
@click.option('--output-file-pattern', type=str)
@click.option('--output-folder', type=click.Path(exists=False))
@click.option('--n-processes', type=int, default=4)
def main(
        tseries_unit: str,
        parcellation_method: str,
        input_folder: pathlib.PosixPath,
        input_file_pattern: str,
        output_file_pattern: str,
        output_folder: pathlib.PosixPath,
        n_processes: int
):
    """
    Calculates GBC values

    Parameters
    ----------
    tseries_unit: str
        Used to avoid potential user CLI mismatch, typo errors where
        for example input files are ptseries but output files are
        saved as ntseries. Specify the unit you want, and the code
        will read and write the correct files for you. Must be one
        of the following:
            * 'd' for dense
            * 'p' for parcel
            * 'n' for network
    parcellation_method: str
        The type of parcellation used, if applicable. Example: 'CABNP'
    input_folder: pathlib.Path
        Path to where you can find all sessions' text files. Assumes
        there aren't session-specific folders
        Example: ~/embarc/inputs
    input_file_pattern: str
        A pattern used to identify the type of input file you would
        like to use. Example:
            'concatenated_rest-Atlas_s_hpss_res-mVWMWB_lpss-CABNP'
    output_file_pattern: str
        Pattern of the output file. Example:
            'concatenated_rest-Atlas_s_hpss_res-mVWMWB_lpss-CABNP'
    output_folder: pathlib.PosixPath
        Where to save the output file. The code creates this folder if
        it doesn't already exist. Example: '~/embarc-gbc'
    n_processes: int
        The number of parallel processes to use. Makes the code
        run a lot faster (minutes vs hours). Example: 20
    """

    # Fail early if there are user typos or invalid arg combinations
    if tseries_unit in ['p', 'n']:
        assert parcellation_method in reference.parcellation.options

    # Create output folder
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Identify relevant file paths
    input_files = []
    for path in Path(input_folder).iterdir():
        if file_match(file_name=path.name, tseries_unit=tseries_unit,
                other_pattern=input_file_pattern,
                parcellation_method=parcellation_method):
            input_files.append(path)

    n_sessions = len(input_files)
    print(f'\nFound {n_sessions} sessions\n')

    # Parallelize CPU-intensive workloads for a big speed boost
    with Pool(n_processes) as pool:

        # Process results in parallel, and show a progress bar via tqdm
        print(f'\ncalculating GBC values')
        gbc_dicts = list(tqdm.tqdm(pool.imap(make_session_row, input_files),
                           total=len(input_files)))

    # Create data frame
    output_data = pd.DataFrame(gbc_dicts)

    # Ensure correct parcel order
    output_data = enforce_parcel_order(
        parcellation_method=parcellation_method, data_frame=output_data)

    # Save to disk
    output_file = f'{output_file_pattern}-{parcellation_method}.{tseries_unit}gbc.csv.gz'
    output_path = output_folder.joinpath(output_file)
    output_data.to_csv(
        path_or_buf=output_path,
        sep=',',
        header=True,
        index=False,
        compression='gzip'
    )


if __name__ == '__main__':
    main()
