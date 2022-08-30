import pathlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List

import click
import pandas as pd
import tqdm

import reference.parcellation
from utils.neuroimage import *


def concatenate_timeseries(
        output_folder: pathlib.PosixPath,
        output_file_pattern: str,
        delimiter: str,
        ordered_input_paths: List[pathlib.PosixPath]
):
    """
    Concatenates two or more time series text files
    for a given session. Pass this function to a process
    pool to parallelize across many sessions

    Parameters
    ----------
    output_folder: pathlib.PosixPath
        Where to save the outputs of this script
        Example: ~/embarc/output
    output_file_pattern: str
        Name of concatenated output file but without the
        session ID because that will be prepended
        automatically. Example:
            concatenated-Atlas_s_hpss_res-mVWMWB_lpss_BOLD
    delimiter: str
        How to separate values in the file, e.g., ',' for CSV
    ordered_input_paths: list[pathlib.PosixPath]
        A pattern used to identify a session's input files
        that you would like to concatenate. For example: [
            UM0121_baseline-rest_run_1-Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0.ptseries.csv
            UM0121_baseline-rest_run_2-Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0.ptseries.csv]
        The order of series in the final concatenated series
        will match the order of series listed, so 1 before
        2 in the example above.
    """

    concatenated_series = pd.concat([pd.read_csv(f) for f in ordered_input_paths], axis=0, ignore_index=True)

    # Save file to disk
    session_id = session_id_from_file_name(ordered_input_paths[0].name)
    parcellation_method = parcellation_from_file_name(ordered_input_paths[0].name)
    tseries_unit = tseries_unit_from_file_name(ordered_input_paths[0].name)
    file_name = f'{session_id}-{output_file_pattern}-{parcellation_method}.{tseries_unit}tseries.csv.gz'
    output_path = pathlib.Path(output_folder).joinpath(file_name)
    concatenated_series.to_csv(
        path_or_buf=output_path,
        sep=delimiter,
        header=True, index=False,
        compression='gzip'
    )


@click.command()
@click.option('--tseries-unit', type=str)
@click.option('--parcellation-method', required=False,
    type=click.Choice(reference.parcellation.options))
@click.option('--input-file-pattern', multiple=True)
@click.option('--input-folder', type=click.Path(exists=True))
@click.option('--output_file_pattern', type=str)
@click.option('--output-folder', type=click.Path())
@click.option('--n-processes', type=int, default=4)
def main(
        tseries_unit: str,
        parcellation_method: str,
        input_file_pattern: List[str],
        input_folder: pathlib.PosixPath,
        output_file_pattern: str,
        output_folder: pathlib.PosixPath,
        n_processes: int
):
    """
    Concatenates two or more time series text files per
    session

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
    input_folder: pathlib.Path
        Path to where you can find all sessions' text files. Assumes
        there aren't session-specific folders
        Example: ~/embarc/inputs
    output_file_pattern: str
        Name of concatenated output file but without the session
        ID because that will be prepended automatically.
        Example: concatenated-Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0
    output_folder: pathlib.PosixPath
        Where to save the outputs of this script
        Example: ~/embarc/output
    n_processes: int
        The number of parallel processes to use. Makes the code
        run a lot faster (minutes vs hours)
        Example: 20
    """

    # Create output folder
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Identify all sessions and relevant file names
    session_files = dict()
    for path in Path(input_folder).iterdir():
        session_id = session_id_from_file_name(path.name)
        for pattern in input_file_pattern:
            if file_match(file_name=path.name, tseries_unit=tseries_unit,
                    other_pattern=pattern,
                    parcellation_method=parcellation_method):
                if session_id in session_files:
                    session_files[session_id][pattern] = path
                else:
                    session_files[session_id] = {pattern: path}
                break

    # Ensure the CLI order of time series is enforced
    ordered_session_files = list()
    for _, mapping in session_files.items():
        ordered = []
        for pattern in input_file_pattern:
            if pattern in mapping:
                path = mapping[pattern]
                ordered.append(path)
        ordered_session_files.append(ordered)

    n_sessions = len(ordered_session_files)
    print(f'\nFound {n_sessions} sessions\n')

    # Parallelize CPU-intensive workloads for a big speed boost
    with Pool(n_processes) as pool:
        """
        Due to how ProcessPoolExecutor is designed, you can't pass the
        parallelized function multiple function arguments unless you
        wrap the function and its arguments in `partial`
        """
        parallelized_function = partial(
            concatenate_timeseries, output_folder, output_file_pattern, ',')

        # Process results in parallel, and show a progress bar via tqdm
        print(f'\nConcatenating timeseries')
        _ = list(tqdm.tqdm(pool.imap(parallelized_function, ordered_session_files),
                           total=len(ordered_session_files)))


if __name__ == '__main__':
    main()
