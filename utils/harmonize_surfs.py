#!/bin/python

import os
import subprocess
from pathlib import Path
from constants import FREESURFER_DIR,RESAMPLE_DIR,RESAMPLE_FS_AVERAGE_DIR,WB_COMMAND,MRIS_CONVERT
#from utilities import templates

# -----------------
# Dictionary with Freesurfer/FSLR reference files
# -----------------
#resample_dir = '../reference/mri_atlas/standard_mesh_atlases/resample_fsaverage'
#fs_dir = '../reference/mri_atlas/Freesurfer7.1.1'   
fs_dir = Path(Path(__file__).parent.resolve(), FREESURFER_DIR)
resample_dir = Path(Path(__file__).parent.resolve(), RESAMPLE_DIR)


# keep track of the mesh resolution for each fsaverage space
fsavg_dict = {
    'fsaverage': '164k',
    'fsaverage6': '41k',
    'fsaverage5': '10k',
    'fsaverage4': '3k'
}

templates = dict()
for fsavg in fsavg_dict.keys():
    templates[fsavg] = dict()

    for hemi_iter in (('lh', 'L'), ('rh', 'R')):
        hemi, HEMI = hemi_iter
        templates[fsavg][hemi] = dict()
        templates[fsavg][hemi]['sphere'] = os.path.join(resample_dir, '{}_std_sphere.{}.{}_fsavg_{}.surf.gii'.format(fsavg, HEMI, fsavg_dict[fsavg], HEMI))
        templates[fsavg][hemi]['area']   = os.path.join(resample_dir, '{}.{}.midthickness_va_avg.{}_fsavg_{}.shape.gii'.format(fsavg, HEMI, fsavg_dict[fsavg], HEMI))
        templates[fsavg][hemi]['white']  = os.path.join(fs_dir, fsavg, 'surf/{}.white'.format(hemi))

# fslr ref files
for fslr_iter in (('fslr164k', '164'), ('fslr32k', '32k'), ('fslr59k', '59k')):
    fslr, fslr_mesh = fslr_iter
    templates[fslr] = dict()

    for hemi_iter in (('lh', 'L'), ('rh', 'R')):
        hemi, HEMI = hemi_iter
        templates[fslr][hemi] = dict()
        templates[fslr][hemi]['sphere'] = os.path.join(resample_dir, 'fs_LR-deformed_to-fsaverage.{}.sphere.{}_fs_LR.surf.gii'.format(HEMI, fslr_mesh))
        templates[fslr][hemi]['area']   = os.path.join(resample_dir, 'fs_LR.{}.midthickness_va_avg.{}_fs_LR.shape.gii'.format(HEMI, fslr_mesh))


def resample_fsavg_to_fslr(gii_input, fslr_mesh, fsavg_mesh, hemi, label_or_metric, out_file):
    '''
    Use HCP Connectome Workbench to resample a surface file from Freesurfer to HCP fslr space.

    Parameters
    ----------
    gii_input: str, path
        Freesurfer surface file in Gifti Format

    fslr_mesh: string
        Mesh resolution in HCP_fslr space. Can be: "164k", "59k", "32k"

    fsavg_mesh: string
        Mesh resolution in fsaverage space. Can be "fsaverage", "fsaverage6", "fsaverage5", "fsaverage4"

    hemi: string
        "lh" or "rh"

    label_or_metric : string
        "label" or "metric". Define the type of surface data being converted. 

    out_file: str, path
        Output file (e.g. my_output.label.gii)
    '''

    # keep track of the mesh resolution for each fsaverage space
    fsavg_dict = {
        'fsaverage': '164k',
        'fsaverage6': '41k',
        'fsaverage5': '10k',
        'fsaverage4': '3k'
    }

    # parse hemisphere
    if hemi == 'lh':
        HEMI='L'
    elif hemi == 'rh':
        HEMI='R'
    else:
        return 

    # parse whether we are resampling a label or 
    if label_or_metric == 'label':
        resample_type = '-label-resample'
    elif label_or_metric == 'metric':
        resample_type = '-metric-resample'
    else: 
        return

    # define relative paths to files
    resample_dir = Path(Path(__file__).parent.resolve(), RESAMPLE_FS_AVERAGE_DIR)
    #resample_dir = '../reference/mri_atlas/standard_mesh_atlases/resample_fsaverage'
    
    orig_sphere = os.path.join(resample_dir, 'fsaverage_std_sphere.{}.{}_fsavg_{}.surf.gii'.format(HEMI, fsavg_dict[fsavg_mesh], HEMI))
    orig_area   = os.path.join(resample_dir, 'fsaverage.{}.midthickness_va_avg.{}_fsavg_{}.shape.gii'.format(HEMI, fsavg_dict[fsavg_mesh], HEMI))
    
    new_sphere = os.path.join(resample_dir, 'fs_LR-deformed_to-fsaverage.{}.sphere.{}_fs_LR.surf.gii'.format(HEMI, fslr_mesh))
    new_area   = os.path.join(resample_dir, 'fs_LR.{}.midthickness_va_avg.{}_fs_LR.shape.gii'.format(HEMI, fslr_mesh))

    # execute
    subprocess.call([
        WB_COMMAND, 
        resample_type,
        gii_input,
        orig_sphere,
        new_sphere,
        'ADAP_BARY_AREA',
        out_file,
        '-area-metrics',
        orig_area,
        new_area
        ])


def wb_resample_label(label_in, label_out, cur_sphere, cur_area, new_sphere, new_area):
    '''
    Wrapper for resampling label file using workbench
    '''
    subprocess.call([WB_COMMAND,
                '-label-resample',
                label_in, 
                cur_sphere,
                new_sphere,
                'ADAP_BARY_AREA',
                label_out,
                '-area-metrics',
                cur_area,
                new_area])


def split_dlabel(cifti_file, gii_left, gii_right):
    '''
    HCP workbench function call to split a dlabel cifti file into cortex only giftis
    '''

    split_dlabel = [WB_COMMAND, '-cifti-separate',cifti_file,'COLUMN', '-label', 'CORTEX_LEFT', gii_left,'-label', 'CORTEX_RIGHT', gii_right]
    subprocess.call([WB_COMMAND,
                '-cifti-separate',
                cifti_file, 
                'COLUMN',
                '-label', 'CORTEX_LEFT', gii_left, 
                '-label', 'CORTEX_RIGHT', gii_right])
    return gii_left, gii_right


def gii_to_annot(gii_file, white_file, annot_file):
    cmd = [
        MRIS_CONVERT,
        '--annot', gii_file, white_file, annot_file
    ]
    subprocess.call(cmd)


def mgh_to_gii(input, output):
    '''
    Wrapper for Freesurfer mris_convert. 

    Convert a Freesurfer surface file (e.g. lh.thickness) to Gifti format

    Parameters
    ----------
    input: string, path
        Full path to the input file

    output: string, path
        Full path to the output file
    '''
    subprocess.call([
        MRIS_CONVERT, input, output
    ])


def aparc_to_gii(gii_file, white_file, aparc_file):
    subprocess.call([
        MRIS_CONVERT,
        '--annot', gii_file, white_file, aparc_file
    ])


def resample_fslr_to_fsavg(gii_input, fslr_mesh, fsavg_mesh, hemi, label_or_metric, out_file):
    '''
    Use HCP Connectome Workbench to resample a surface atlas from HCP

    @param gii_input: GiFTI formatted surface input file (freesurfer space)
    @param fslr_mesh: Mesh resolution (32k, 59k, 164k) in fslr space to resample to
    @param fsavg_mesh: Resolution of the freesurfer input gifti file (fsaverage, fsaverage6, fsaverage5, etc...)
    @param hemi: "lh" or "rh"
    @label_or_metric: "label" or "metric"
    @out_file: full path to the output file (e.g. my_output.label.gii)
    '''

    # keep track of the mesh resolution for each fsaverage space
    fsavg_dict = {
        'fsaverage': '164k',
        'fsaverage6': '41k',
        'fsaverage5': '10k',
        'fsaverage4': '3k'
    }

    # parse hemisphere
    if hemi == 'lh':
        HEMI='L'
    elif hemi == 'rh':
        HEMI='R'
    else:
        return 

    # parse whether we are resampling a label or 
    if label_or_metric == 'label':
        resample_type = '-label-resample'
    elif label_or_metric == 'metric':
        resample_type = '-metric-resample'
    else: 
        return

    # define relative paths to files
    #resample_dir = '../reference/mri_atlas/standard_mesh_atlases/resample_fsaverage'
    resample_dir = RESAMPLE_DIR
    #resample_dir = Path(Path(__file__).parent.resolve(), '../reference/mri_atlas/standard_mesh_atlases/resample_fsaverage')

    new_sphere = os.path.join(resample_dir, 'fsaverage_std_sphere.{}.{}_fsavg_{}.surf.gii'.format(HEMI, fsavg_dict[fsavg_mesh], HEMI))
    new_area   = os.path.join(resample_dir, 'fsaverage.{}.midthickness_va_avg.{}_fsavg_{}.shape.gii'.format(HEMI, fsavg_dict[fsavg_mesh], HEMI))
    
    orig_sphere = os.path.join(resample_dir, 'fs_LR-deformed_to-fsaverage.{}.sphere.{}_fs_LR.surf.gii'.format(HEMI, fslr_mesh))
    orig_area   = os.path.join(resample_dir, 'fs_LR.{}.midthickness_va_avg.{}_fs_LR.shape.gii'.format(HEMI, fslr_mesh))

    # execute
    wb_cmd = [
        WB_COMMAND, 
        resample_type,
        gii_input,
        orig_sphere,
        new_sphere,
        'ADAP_BARY_AREA',
        out_file,
        '-area-metrics',
        orig_area,
        new_area
        ]
    subprocess.call(wb_cmd)

