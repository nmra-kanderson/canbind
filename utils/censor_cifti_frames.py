import glob
import nibabel as nb
import shutil
import pathlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List
import tqdm
import click

from utils.neuroimage import parse_session_hcp, load_motion_scrub, get_qunex_dirs


@click.command()
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-folder')
@click.option('--regex-session-filter', default=None)
@click.option('--study-name', default=None)
@click.option('--scan-types', default=None, multiple=True)
@click.option('--n-processes', type=int, default=4)
@click.option('--input-file-pattern', multiple=False)

def main(
        sessions_dir: pathlib.PosixPath,
        regex_session_filter: str,
        output_folder: str,
        study_name: str,
        n_processes: int,
        scan_types: List[str],
        input_file_pattern: str
        ):
    """
    Censor high motion frames from CIFTI file
    """

    # Create output folder if it doesn't exist
    output_folder       = Path(output_folder)
    study_output_folder = Path(output_folder, study_name)
    output_folder.mkdir(parents=True, exist_ok=True)
    study_output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of QuNex subdirectories for processing
    session_dirs = get_qunex_dirs(sessions_dir, regex_session_filter)
    session_dir = session_dirs[0]

    for scan in scan_types:
        print(f'Processing Scan: {scan}')

        # Parallelize workloads for a big speed boost
        with Pool(n_processes) as pool:

            parallelized_function = partial(
                censor_cifti,
                input_file_pattern, study_output_folder, scan, study_name)

            # Process results in parallel, and show a progress bar via tqdm
            _ = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                               total=len(session_dirs)))


def censor_cifti(input_file_pattern, study_output_folder, scan, study_name, session_dir):

    # create subjects specific output directory in the {}/raw subfolder
    session_id = session_dir.stem
    session_output_folder = Path(study_output_folder, 'raw', session_id, 'functional')
    session_output_folder.mkdir(exist_ok=True, parents=True)

    # read scan number to name mapping
    scan_dict = parse_session_hcp(Path(session_dir, 'session_hcp.txt'), [scan])
    
    # parcellate each scan 
    for scan_name in scan_dict.keys():
        scan_num    = scan_dict[scan_name]
        scan_string = scan_name.replace('bold','').replace(' ','').replace('-','_')

        # get the dtseries file for processing 
        dtseries_glob = glob.glob(str(Path(session_dir, 'images/functional', f'*{scan_num}*{str(input_file_pattern)}')))
        if len(dtseries_glob) != 1:
            continue
        dtseries_file = dtseries_glob[0]
        
        # file info
        file_descriptor = '_'.join(dtseries_file.split('/')[-1].split('_')[1:])
        file_descriptor = file_descriptor.split('.dtseries')[0]

        # define path to the destination dtseries file
        dtseries_dest_fname = f'sub-{session_id}_task-{scan_string}_{file_descriptor}_study-{study_name}.dtseries.nii'
        dtseries_dest       = Path(session_output_folder, dtseries_dest_fname)
        shutil.copy(dtseries_file, dtseries_dest)

        # censor file
        dtseries_censor_dest_fname = f'sub-{session_id}_task-{scan_string}_{file_descriptor}_FrameCensored_study-{study_name}.dtseries.nii'
        dtseries_censor_dest       = Path(session_output_folder, dtseries_censor_dest_fname)

        # read motion scrub information
        censor_frames = load_motion_scrub(session_dir, f'bold{scan_num}')

        # wrapper for wb_command
        censor_dtseries(cii_in=dtseries_file, 
                        censor_frames=censor_frames, 
                        cii_out=dtseries_censor_dest)


def censor_dtseries(cii_in, censor_frames, cii_out):

    cii_obj = nb.load(cii_in)
    cii_dat = cii_obj.get_fdata()
    censor_dat = cii_dat[censor_frames == 1,]
    cii_hdr    = cii_obj.header
    
    num_tr, num_vertices = cii_obj.dataobj.shape
    assert len(censor_frames) == num_tr

    ax_0 = nb.cifti2.SeriesAxis(start=0, step=cii_hdr.get_index_map(0).series_step, size=censor_dat.shape[0]) 
    ax_1 = cii_hdr.get_axis(1)
    new_h = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))

    censor_cii = nb.cifti2.cifti2.Cifti2Image(dataobj=censor_dat, header=new_h)
    censor_cii.to_filename(cii_out)


if __name__ == '__main__':
    main()





