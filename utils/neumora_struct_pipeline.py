import os
import subprocess
import tempfile
import traceback
from random import randint
from pathlib import Path
from typing import List

from functools import partial
from multiprocessing import Pool
import pandas as pd
import nibabel as nb
import tqdm
import click
import quilt3 
import IPython
from PIL import ImageColor

from neuroimage import get_qunex_dirs
from harmonize_surfs import resample_fslr_to_fsavg, split_dlabel
from constants import FREESURFER_AVERAGE_DIR,QUILT_REF_PACKAGE,TRESTLE_REGISTRY,LH_WHITE_PATH,RH_WHITE_PATH,MRIS_CONVERT,MRI_SURF2SURF,MRI_ANATOMICAL_STATS,FREESURFER_HOME


# run this command before importing any local functions
def download_reference():
    # check reference files
    file_loc = __file__
    parent_dir = Path(file_loc).parent.resolve()
    ref_dir    =  Path(parent_dir, 'reference')
    if not ref_dir.exists():
        p = quilt3.Package.browse(
                    QUILT_REF_PACKAGE,
                    TRESTLE_REGISTRY
                )
        p.fetch(dest=parent_dir)
download_reference()



@click.command()
@click.option('--sessions-dir',type=click.Path(exists=True), envvar='SESSIONS_DIR')
@click.option('--output-dir',envvar='OUTPUT_DIR')
@click.option('--regex-session-filter', default=None,envvar='REGEX_SESSION_DIR')
@click.option('--study-name', type=str,envvar='STUDY_NAME')
@click.option('--cifti-atlas', type=str,envvar='CIFTI_ATLAS')
@click.option('--cifti-atlas-name', type=str, default='Desikan',envvar='CIFTI_ATLAS_NAME')
@click.option('--n-processes', type=int, default=4,envvar='N_PROCESSES')
@click.option('--overwrite', type=bool, default=True,envvar='OVERWRITE')
def main(
        sessions_dir: str,
        output_dir: str,
        regex_session_filter: str,
        study_name: str,
        cifti_atlas: str,
        cifti_atlas_name: str,
        n_processes: int,
        overwrite: bool
        ):
    """
    Compile Freesurfer estimates of brain anatomy for 
    a given QuNex output directory (e.g. GrayVol, ThickAvg, 
    SurfArea). Data will be saved into three directories: 
    "raw", "refined", and "production" to make for easy 
    ingestion into the Trestle ecosystem. By default, Freesurfer
    estimates will be in the default "Desikan" atlas space. 
    Users may pass a CIFTI atlas (in fslr32k HCP space) to 
    get anatomical estimates for each constituent cortical parcel. 
    Sub-cortical anatomical estimates will also be summarized. 

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
    overwrite: bool
        Overwrite everything fresh 
    """

    print(f'Sessions Dir:          {sessions_dir}')
    print(f'Output Dir:            {output_dir}')
    print(f'Regex Session Filter:  {regex_session_filter}')
    print(f'Study Name:            {study_name}')
    print(f'Cifti Atlas:           {cifti_atlas}')
    print(f'Cifti Atlas Name:      {cifti_atlas_name}')
    print(f'N Processes:           {n_processes}')
    
    # Create output folder if it doesn't exist
    output_dir = Path(output_dir)
    study_output_dir = Path(output_dir, study_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of QuNex subdirectories for processing
    session_dirs = get_qunex_dirs(sessions_dir, regex_session_filter)

    # If compiling data according to a non-Desikan/non-FS default atlas,
    # then we need to resample the parcellation to each individual.
    # So we convert the fslr32k cifti to lh/rh annot files and then 
    # project into fsaverage template space
    # --------------------
    if cifti_atlas is not None:
        annot_fsavg_left, annot_fsavg_right = convert_cifti_ctx_to_fsavg(cifti_atlas)
    else:
        annot_fsavg_left = None
        annot_fsavg_right = None
        
    # Path to temp dir
    tmp_dir = tempfile.gettempdir()

    # copy the fsaverage brain to the tmp dir
    file_loc = __file__
    # file_loc = Path('/imaging-features/utils/neumora_struct_pipeline.py')

    # check reference files
    parent_dir = Path(file_loc).parent.resolve()
    ref_dir    =  Path(parent_dir, 'reference')
    if not ref_dir.exists():
        p = quilt3.Package.browse(
                    QUILT_REF_PACKAGE,
                    TRESTLE_REGISTRY
                )
        p.fetch(dest=parent_dir)

    fsavg_dir  = Path(parent_dir, FREESURFER_AVERAGE_DIR)
    if Path(tmp_dir, 'fsaverage').exists():
        os.remove(Path(tmp_dir, 'fsaverage'))
    os.symlink(fsavg_dir, Path(tmp_dir, 'fsaverage'))


    # Compile Raw Freesurfer anatomical measurements
    # --------------------
    print_feedback('Compiling "Raw" Freesurfer anatomical estimates')
    with Pool(n_processes) as pool:
        # pass inputs to the "compile_freesurfer_single_session" function
        parallelized_function = partial(
            compile_freesurfer_single_session,
            study_output_dir,
            study_name,
            cifti_atlas_name,
            annot_fsavg_left,
            annot_fsavg_right,
            overwrite)

        # Process results in parallel, and show a progress bar via tqdm
        print('\nCompiling Freesurfer Features')
        fs_data_list = list(tqdm.tqdm(pool.imap(parallelized_function, session_dirs),
                        total=len(session_dirs)))

    # get rid of failed session
    fs_data_list = [dat for dat in fs_data_list if dat is not None]
    fs_df        = pd.concat(fs_data_list)

    # save production-ready/concatenated dataframes
    # -------------------
    production_dir = Path(study_output_dir, 'production', 'anatomical')
    production_dir.mkdir(exist_ok=True, parents=True)

    for modality in ['GrayVol', 'ThickAvg', 'SurfArea', 'Global']:
        # path to the production data
        modality_fname = f'study-{study_name}_freesurfer_{modality}_atlas-{cifti_atlas_name}.csv.gz'
        modality_file  = Path(production_dir, modality_fname)

        # subset dataframe to the current modality
        if modality == 'Global':
            modality_cols = list(fs_df.columns[~fs_df.columns.str.contains('GrayVol|ThickAvg|SurfArea')])
        else:
            modality_cols = ['id'] + list(fs_df.columns[fs_df.columns.str.contains(modality)])
        modality_df = fs_df[modality_cols]
        
        print('Saving compiled "Production" Freesurfer data: ')
        print(f'\t{modality_file}')
        # save to disk 
        modality_df.to_csv(modality_file, index=None, compression='gzip')


def compile_freesurfer_single_session(study_output_dir,
                                    study_name,
                                    cifti_atlas_name,
                                    annot_fsavg_left,
                                    annot_fsavg_right,
                                    overwrite,
                                    session_dir):
    """
    Retrieve Freesurfer volumetric/surface statistics. By default,
    these data will be ROI stats in the default Desikan atlas space.
    However, other ROI sets/atlases can be passed in by specifying the
    full path to the freesurfer annotation data (annot_fsavg_left/annot_fsavg_right)

    Parameters
    ----------
    modality: str
        Tthe Freesurfer measurement type to compile.
        "GrayVol", "ThickAvg", "SurfArea", "Global"
        Can be one of:
            GrayVol:  Volume of Cortical Gray Matter
            ThickAvg: Mean Thickness of Cortical Gray Matter
            SurfArea: Surface Area of Cortical Gray Matter
            Global:   Whole-brain or Whole-hemisphere estimates (e.g. size of ventricles)
    study_output_dir: str
        Path to the study-specific data output directory. 
            e.g. "/fmri-qunex/research/imaging/datasets/embarc/qunex_features/EMBARC"
    study_name: str
        Name of the study being processed.
            e.g. "EMBARC"
    cifti_atlas_name: str
        Name of the cortical atlas. 
            e.g. "Desikan", "CABNP"
    annot_fsavg_left: None, str
        Optional argument. If specified, full path to the LH cortical fsaverage annot file.
    annot_fsavg_right: None, str
        Optional argument. If specified, full path to the RH cortical fsaverage annot file.
    session_dir: str
        Full path to the session directory with session output.
    overwrite: bool
        Fresh and clean overwrite
    """

    try:
        # create subject specific output directory in the "{output_dir}/raw" subfolder
        session_id = session_dir.stem
        session_output_folder = Path(study_output_dir, 'raw', session_id, 'anatomical')
        session_output_folder.mkdir(exist_ok=True, parents=True)
        
        refined_output_folder = Path(study_output_dir, 'refined', session_id, 'anatomical')
        refined_output_folder.mkdir(exist_ok=True, parents=True)

        # check freesurfer symlinks
        session_fs_dir = Path(session_dir, 'hcp', session_id, 'T1w', session_id)
        os.system(f'chmod -R 777 {session_fs_dir}')
        for hemi in ['lh', 'rh']:
            for surf in ['white', 'pial']:
                fs_file = Path(session_fs_dir, f'surf/{hemi}.{surf}')            
                if not fs_file.is_symlink():
                    src = Path(session_fs_dir, f'surf/{hemi}.{surf}.rawavg.conf')  
                    # if file is very small
                    if fs_file.exists():
                        if os.path.getsize(fs_file) < 100:
                            os.remove(fs_file)
                            os.symlink(src, fs_file)
                    else:
                        os.symlink(src, fs_file)

        # if we have fed in a custom parcellation, compute the anatomical stats
        if annot_fsavg_left is not None and annot_fsavg_right is not None:

            # symlink the freesurfer data from QuNex to tmp working dir
            copy_qunex_freesurfer_to_tmpdir(session_dir, session_id)

            # resample the custom parcellation to individual freesurfer space
            # then get the anatomical estimates (i.e. freesurfer .stats file)
            lh_stats, rh_stats = compute_stats_by_parcellation(session_output_folder,
                                                                session_id,
                                                                annot_fsavg_left,
                                                                annot_fsavg_right,
                                                                overwrite)
        else:
            # otheriwse, use the desikan default stats 
            lh_stats = 'lh.aparc.stats'
            rh_stats = 'rh.aparc.stats'

        # read thickness/volume estimates
        try:
            sesh_data = read_freesurfer(session_dir=session_dir, stat_files=['aseg.stats', lh_stats, rh_stats])
            return sesh_data
        except:
            print(f'Freesurfer processing failed for session: {session_dir.stem}')
    except Exception:
        traceback.print_exc()
                
    #    except:
    #        print(f'Freesurfer processing failed for session: {session_dir.stem}')
    #    else:
    #        # split data by modality, and save individual data to file
    #        csv_gz=f'sub-{session_id}_study-{study_name}_atlas-{cifti_atlas_name}_freesurfer_{modality}.csv.gz'
    #        modality_df = write_freesurfer(fs_df=sesh_data,
    #                                        modality=modality,
    #                                        csv_gz=csv_gz,
    #                                        out_dir=refined_output_folder)
    #        return modality_df
    #except Exception:
    #    traceback.print_exc()


def uniqify_labels(label_gii):
    """
    Make sure that each label has a unique color associated with it.
    This fixes a compatablity issue when converting from a label.gii to annot file

    Parameters
    -----------
    label_gii: str
        Full path to the .label.gii file
    """
    # load label file 
    label_obj = nb.load(label_gii)

    # get array/dict of labels and colors
    label_dict   = label_obj.labeltable.get_labels_as_dict()
    label_colors = [(l.red, l.green, l.blue) for l in label_obj.labeltable.labels]

    # if there are more labels than colors, replace the color of each parcel with a new unique color
    if len(label_dict) is not len(set(label_colors)):
        print('Uniqifying colors in the label.gii file')
        # create new list of unique color hexs
        color = []
        n = len(label_dict) + 100
        for i in range(n):
            color.append('#%06X' % randint(0, 0xFFFFFF))
            color = list(set(color))

        # iterate over each parcel and plug in the new label color
        for i in range(len(label_obj.labeltable.labels)):
            label_obj.labeltable.labels[i].red =  ImageColor.getcolor(color[i], "RGB")[0]/255
            label_obj.labeltable.labels[i].green =  ImageColor.getcolor(color[i], "RGB")[1]/255
            label_obj.labeltable.labels[i].blue =  ImageColor.getcolor(color[i], "RGB")[2]/255
        # save the uniqueified label.gii file
        nb.save(label_obj, label_gii)


def convert_cifti_ctx_to_fsavg(cifti_atlas):
    """
    Take a fslr32k CIFTI dlabel file as input,
    resample to fsaverage template. 
    Return two (lh/rh) annot files in fsaverage space.
    All work is done in a temporary directory.

    Parameters
    ----------
    cifti_atlas: str
        Full path to the CIFTI dlabel file. Must be in fslr32k atlas space.
    
    Returns
    ----------
    annot_fsavg_left: str
        Full path to the lh fsaverage annot file in the tmp directory
    annot_fsavg_right: str
        Full path to the rh fsaverage annot file in the tmp directory
    """
    # split
    tmp_dir        = tempfile.gettempdir()
    os.environ["SUBJECTS_DIR"] = tmp_dir
    dlabel_fname   = cifti_atlas.split('/')[-1]

    # split cifti into lh/rh gifti label files
    print_feedback('Splitting CIFTI cortex data into LH/RH label.gii files')
    gii_left     = Path(tmp_dir, dlabel_fname.replace('.dlabel.nii', '_lh.label.gii'))
    gii_right    = Path(tmp_dir, dlabel_fname.replace('.dlabel.nii', '_rh.label.gii'))
    gii_left, gii_right = split_dlabel(cifti_file=cifti_atlas, gii_left=gii_left, gii_right=gii_right)
    print(gii_left)
    print(gii_right)

    # LH fslr to fsavg
    print_feedback('Resample label file from FSLR32k to fsaverage')
    gii_fsavg_left = Path(tmp_dir, dlabel_fname.replace('.dlabel.nii', '_fsaverage_lh.label.gii'))
    resample_fslr_to_fsavg(gii_input=gii_left, fslr_mesh='32k', fsavg_mesh='fsaverage', 
                            hemi='lh', label_or_metric='label', out_file=gii_fsavg_left)
    uniqify_labels(gii_fsavg_left)
    print(gii_fsavg_left)
    lh_gii = nb.load(gii_fsavg_left)


    # RH fslr to fsavg
    gii_fsavg_right = Path(tmp_dir, dlabel_fname.replace('.dlabel.nii', '_fsaverage_rh.label.gii'))
    resample_fslr_to_fsavg(gii_input=gii_right, fslr_mesh='32k', fsavg_mesh='fsaverage', 
                            hemi='rh', label_or_metric='label', out_file=gii_fsavg_right)    
    uniqify_labels(gii_fsavg_right)      
    print(gii_fsavg_right)

    # create fsaverage annot file 
    print_feedback('Convert label.gii to annot format (in fsaverage space)')
   
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME
    # lh
    annot_fsavg_left = Path(tmp_dir, dlabel_fname.replace('.dlabel.nii', '_fsaverage_lh.annot'))
    lh_white         = LH_WHITE_PATH
    lh_annot_cmd     = [MRIS_CONVERT, '--annot', gii_fsavg_left, lh_white, annot_fsavg_left]
    print(lh_annot_cmd)
    subprocess.call(lh_annot_cmd)

    # rh
    annot_fsavg_right = Path(tmp_dir, dlabel_fname.replace('.dlabel.nii', '_fsaverage_rh.annot'))
    rh_white         = RH_WHITE_PATH
    rh_annot_cmd     = [MRIS_CONVERT, '--annot', gii_fsavg_right, rh_white, annot_fsavg_right]
    print(rh_annot_cmd)
    subprocess.call(rh_annot_cmd)
    
    return annot_fsavg_left, annot_fsavg_right


def copy_qunex_freesurfer_to_tmpdir(session_dir, session_id):
    """
    Copy freesurfer data from QuNex directory to tmp_dir

    Parameters
    ----------
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    session_id: str
        Session ID
    """

    # Path to temp dir
    tmp_dir = tempfile.gettempdir()

    # Path to the session Freesurfer output in QuNex
    orig_subjects_dir = Path(session_dir, 'hcp', session_id, 'T1w', session_id)
    dest_dir = Path(tmp_dir, session_id)    
    # symlink the QuNex data to the tmp_dir
    if not Path(tmp_dir, session_id).exists():
        dest_dir = Path(tmp_dir, session_id)
        os.symlink(orig_subjects_dir, dest_dir)
    # open permissions, otherwise can't create new data
    #os.system(f'sudo chmod 777 -R {Path(tmp_dir, session_id)}')


def compute_stats_by_parcellation(session_output_folder,
                                    session_id,
                                    annot_fsavg_left,
                                    annot_fsavg_right,
                                    overwrite):
    """
    Compute cortical freesurfer estimates for a custom parcellation. 

    Parameters
    ----------
    session_dir: str
        Full path to the QuNex session output
    session_output_folder: str
        Folder to output the anatomical stats
        Example: {output_dir}/{study_name}/raw/{session_id}/anatomical
    session_dir: str
        Session ID
        Example: TX0031_baseline
    annot_fsavg_left: str
        Path to the custom parcellation, saved as a freesurfer annot file
        in faverage space (LEFT Hemisphere)
    annot_fsavg_right: str
        Path to the custom parcellation, saved as a freesurfer annot file
        in faverage space (RIGHT Hemisphere)

    Returns:
    ----------
    str
        Path to the computed LEFT hemisphere freesurfer stats file
    str
        Path to the computed RIGHT hemisphere freesurfer stats file
    """

    # resample the surface annot file from fsaverage to individual space
    stats_file_dict = dict()
    for hemi, annot_fsavg in (('lh', annot_fsavg_left), ('rh', annot_fsavg_right)):

        # do the fsaverage >> individual transform of annot file
        indiv_annot = Path(session_output_folder, f'{session_id}_{annot_fsavg.stem}.annot')
        if not indiv_annot.exists() or overwrite:
            os.environ['SUBJECTS_DIR'] = tempfile.gettempdir()
            surf2surf_cmd = [
                'mri_surf2surf', '--hemi', hemi, '--srcsubject', 'fsaverage',
                '--sval-annot', annot_fsavg,
                '--trgsubject', session_id, '--tval', indiv_annot
                ]
            #print(surf2surf_cmd)
            subprocess.call(surf2surf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # use indiv annot file to compute freesurfer anatomical stats
        aparc_stats = Path(session_output_folder, f'{session_id}_{annot_fsavg.stem}.aparc.stats')
        if not aparc_stats.exists() or overwrite:
            os.environ['SUBJECTS_DIR'] = tempfile.gettempdir()
            mri_anat_stats_cmd = [
                'mris_anatomical_stats', '-f', aparc_stats,
                '-a', indiv_annot,
                session_id, hemi
            ]
            #print(mri_anat_stats_cmd)
            subprocess.call(mri_anat_stats_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        stats_file_dict[hemi] = aparc_stats
    return stats_file_dict['lh'], stats_file_dict['rh']


def write_freesurfer(fs_df, modality, csv_gz, out_dir):
    """
    Subset a dataframe of Freesurfer measures to a particular modality.
    Write the data for an individual subject to file.
    
    Parameters
    ----------
    fs_df: pd.DataFrame
        Data frame for an individual subject that contains 
        thickness/volume/area/global Freesurfer measures
    modality: str
        ThickAvg, SurfArea, GrayVol, Global
        The Freesurfer modality to subset and write
    csv_gz: str
        Name of the individual Freesurfer data file to write
    out_dir: str
        Directory to write the "csv_gz" file

    Returns
    ----------
    pd.DataFrame
        Freesurfer data for an individual, subsetted to the "modalitY"
    """
    # subset dataframe to the current modality
    if modality == 'Global':
        modality_cols = list(fs_df.columns[~fs_df.columns.str.contains('GrayVol|ThickAvg|SurfArea')])
    else:
        modality_cols = ['id'] + list(fs_df.columns[fs_df.columns.str.contains(modality)])
    modality_df  = fs_df[modality_cols]

    # save individual data to file
    modality_file = Path(out_dir, csv_gz)
    modality_df.to_csv(modality_file, index=None, compression='gzip')
    return modality_df


def conv_lines_to_df(fs_data, session_id):
    '''
    Convert Freesurfer data to a dataframe

    Parameters
    ----------
    fs_data : List[str]
        Freesurfer data stored.
    session_id : str
        Session ID.
    
    Returns
    ----------
    pd.DataFrame
    '''
    # read freesurfer data
    fs_hdrs = [x for x in fs_data if 'ColHeaders' in x][0].split()
    fs_hdrs = fs_hdrs[2:]
    fs_df   = pd.DataFrame([x.split() for x in fs_data if '#' not in x])
    fs_df.columns = fs_hdrs
    fs_df.insert(0, 'id', session_id)
    return fs_df


def process_surface(fs_data, session_id):
    '''
    Process Freesurfer surface data

    Parameters
    ----------
    fs_data : List[str]
        Freesurfer data stored.
    session_id : str
        Session ID.
    
    Returns
    ----------
    pd.DataFrame
    '''
    # read freesurfer data
    hemi  = [x.split('hemi')[1].replace('\n','').replace(' ','') for x in fs_data if 'hemi' in x][0]
    fs_df = conv_lines_to_df(fs_data, session_id)

    # Convert Long-to-Wide
    meas_list = []
    for meas in ['SurfArea', 'ThickAvg', 'GrayVol']:
        fs_wide = fs_df.pivot(columns='StructName', values=meas, index='id')
        fs_wide.columns = meas + '_' + hemi + '_' +  fs_wide.columns
        meas_list.append(fs_wide)
    meas_df = pd.concat(meas_list, axis=1)
    return meas_df


def process_volume(fs_data, session_id):
    '''
    Process Freesurfer volumetric data

    Parameters
    ----------
    fs_data : List[str]
        Freesurfer data stored.
    session_id : str
        Session ID.
    
    Returns
    ----------
    pd.DataFrame
    '''
    fs_df   = conv_lines_to_df(fs_data, session_id)
    fs_wide = fs_df.pivot(columns='StructName', values='Volume_mm3', index='id')
    return fs_wide


def read_freesurfer_header(fs_dir, stat_file='aseg.stats'):
    """
    Read data stored within the header of a Freesurfer stats file

    Parameters
    ----------
    fs_dir : pathlib.PosixPath
        Absolute path to the freesurfer output directory.
    stat_file : str
        One of the freesurfer *.stats file. e.g. aseg.stats.

    Return
    ----------
    pd.DataFrame
    """
    # path to the freesurfer stats file
    fs_path = Path(fs_dir, 'stats', stat_file)
    
    try:
        with open(fs_path, 'r') as f:
            fs_data = f.readlines()

        # Each line with data starts with "# Measure"
        measure_lines = [x for x in fs_data if '# Measure' in x]
        # list with the name of each measure
        measure_list = []
        for x in measure_lines:
            x_split = x.split()[2].replace(',','')
            if x_split == 'Cortex':
                x_str = x.split()[3].replace(',','')
                measure_list.append('{}_{}'.format(x_split, x_str))
            else:
                measure_list.append(x_split)
        #measure_list  = [x.split()[2].replace(',','') for x in measure_lines]
        # list with the actual measurement values
        value_list    = [float(x.split()[-2].replace(',','')) for x in measure_lines]
        
        # format into dataframe and return
        aseg_row         = pd.DataFrame(value_list).transpose()
        aseg_row.columns = measure_list
        aseg_row.insert(0, 'id', fs_dir.stem)
        aseg_row.set_index(['id'], inplace=True)
        return aseg_row
        
    except FileNotFoundError: 
        print(f'Could not read freesurfer header for {fs_dir.stem} for session: {str(fs_path)}')


def read_freesurfer(session_dir, stat_files=['aseg.stats', 'lh.aparc.stats', 'rh.aparc.stats']):
    '''
    Read freesurfer stats from a QuNex formatted directory.

    Parameters
    ----------
    session_dir: str/path
        Full path to the QuNex session output
    stat_files: list[str]
        List of strings defining the Freesurfer stats files to read, e.g.
            lh.curv.stats
            rh.curv.stats
            lh.aparc.stats
            rh.aparc.stats
            aseg.stats

    Returns
    ----------
    pd.DataFrame
    '''
    # path to freesurfer output
    fs_dir = Path(session_dir, 'hcp', session_dir.stem, 'T1w', session_dir.stem)
    
    # read data stored in the header of stats files
    hdr_measures = read_freesurfer_header(fs_dir, 'aseg.stats')
    hdr_surf_measures = read_freesurfer_header(fs_dir, 'lh.aparc.stats')
    hdr_cortex   = hdr_surf_measures.loc[:,~hdr_surf_measures.columns.isin(hdr_measures.columns)]
    hdr_measures = pd.concat([hdr_measures, hdr_cortex], axis=1)

    # Read each freesurfer stats file
    fs_list = [hdr_measures]
    for stat_file in stat_files:
        if not Path(stat_file).exists():
            fs_path = Path(fs_dir, 'stats', stat_file)
        else: 
            fs_path = stat_file
        try:
            with open(fs_path, 'r') as f:
                fs_data = f.readlines()
        except FileNotFoundError:
            print('Cannot find: {}'.format(fs_path))
            continue  

        # infer the data type (surface or volume)
        anat_type = [x.split('anatomy_type')[1].replace('\n','').replace(' ','') for x in fs_data if 'anatomy_type' in x][0]

        # Process the data, which is stored as a list of strings
        # surface/volumetric data need to be processed differently
        if anat_type == 'surface':
            fs_df = process_surface(fs_data, session_dir.stem)
        elif anat_type == 'volume':
            fs_df = process_volume(fs_data, session_dir.stem)
        # append data to larger list
        fs_list.append(fs_df)

    freesurf_df = pd.concat(fs_list, axis=1)
    freesurf_df.insert(0, 'id', freesurf_df.index)
    return freesurf_df


def print_feedback(print_string):
    """
    Print fancy feedback
    """
    print('------------------------------')
    print(print_string)
    print('------------------------------')


if __name__ == '__main__':
    main()