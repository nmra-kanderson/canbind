import click
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import tqdm
import traceback
from utils.neuroimage import scrub_motion, parse_session_hcp
import pickle


# TODO: Add docstring
def check_file(scans, session_hcp_path):
    """
    I run these in a process pool as opposed to a thread pool
    for maximum parallelization, but that means the processes
    in the pool can't update a results dict in parallel, so I
    return a small dict, and then aggregate the dicts after the
    pool has been exhausted of tasks
    """
    outcomes = {'session_hcp_path': session_hcp_path}
    if not Path(session_hcp_path).exists():
        outcomes['session_hcp_file_not_found'] = None
    else:
        try:
            outcomes['mapping'] = parse_session_hcp(file_path=session_hcp_path, scan_types=scans)
        except Exception as exception:
            outcomes['other'] = traceback.format_exc()
    return outcomes


# TODO: Add docstring
@click.command()
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-dir', type=str)
@click.option('--scans', multiple=True)
@click.option('--n-processes', type=int, default=100)
def main(sessions_dir, output_dir, scans, n_processes):

    # Create output folder if it doesn't exist
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Identify all session directories
    session_hcp_paths = list()
    for session_path in Path(sessions_dir).iterdir():
        if '_' in session_path.stem:
            session_hcp_path = Path(session_path).joinpath('session_hcp.txt')
            session_hcp_paths.append(session_hcp_path)
    all_sessions = set(map(lambda sd: sd.stem.split("_")[0], session_hcp_paths))
    n_sessions = len(all_sessions)
    print(f'Found {n_sessions} sessions')

    # Contains BOLD IDs of each of the scan types
    mapping = dict()

    # Process each of the scan types
    for scan in scans:
        # Parallelize CPU-intensive workloads for a big speed boost
        with Pool(n_processes) as pool:
            """
            Due to how ProcessPoolExecutor is designed, you can't pass the
            parallelized function multiple function arguments unless you
            wrap the function and its arguments in `partial`
            """
            parallelized_function = partial(check_file, scans)

            # Process results in parallel, and show a progress bar via tqdm
            print(f'Processing scans of type: {scan}')
            results = list(tqdm.tqdm(pool.imap(parallelized_function, session_hcp_paths), total=len(session_hcp_paths)))

            fout = Path(output_folder).joinpath(f'results.pickle')
            with open(str(fout), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
