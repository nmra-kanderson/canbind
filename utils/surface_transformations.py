import os
import subprocess


def parse_hemi(hemisphere):
    if hemisphere == 'lh' or hemisphere == 'L':
        HEMI='L'
        hemi='lh'
    if hemisphere == 'lh' or hemisphere == 'L':
        HEMI='R'
        hemi='rh'
    return hemi, HEMI
 

def resample_fslr_to_fsindiv(gii_input, fslr_mesh, fsavg_mesh, hemi, label_or_metric, out_file):
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
    fsavg_res = fsavg_dict[fsavg_mesh]

    # parse hemisphere
    hemi, HEMI = parse_hemi(hemi)

    # parse whether we are resampling a label or 
    if label_or_metric == 'label':
        resample_type = '-label-resample'
    elif label_or_metric == 'metric':
        resample_type = '-metric-resample'
    else: 
        return

    # define relative paths to files
    #__file__ = '/Users/kevin.anderson/Projects/imaging-features'
    resample_dir = Path(__file__).parent.parent.joinpath('reference/standard_mesh_atlases/resample_fsaverage')
    
    orig_sphere_file = f'fsaverage_std_sphere.{HEMI}.{fsavg_res}_fsavg_{HEMI}.surf.gii'
    orig_sphere = os.path.join(resample_dir, orig_sphere_file)

    orig_area_file = f'fsaverage.{HEMI}.midthickness_va_avg.{fsavg_res}_fsavg_{HEMI}.shape.gii'
    orig_area   = os.path.join(resample_dir, orig_area_file)
    
    new_sphere_file = f'fs_LR-deformed_to-fsaverage.{HEMI}.sphere.{fslr_mesh}_fs_LR.surf.gii'
    new_sphere = os.path.join(resample_dir, new_sphere_file)

    new_area_file = f'fs_LR.{HEMI}.midthickness_va_avg.{fslr_mesh}_fs_LR.shape.gii'
    new_area   = os.path.join(resample_dir, 'fs_LR.{}.midthickness_va_avg.{}_fs_LR.shape.gii')

    # execute
    subprocess.call([
        'wb_command', 
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
    fsavg_res = fsavg_dict[fsavg_mesh]

    # parse hemisphere
    hemi, HEMI = parse_hemi(hemi)

    # parse whether we are resampling a label or 
    if label_or_metric == 'label':
        resample_type = '-label-resample'
    elif label_or_metric == 'metric':
        resample_type = '-metric-resample'
    else: 
        return

    # define relative paths to files
    #__file__ = '/Users/kevin.anderson/Projects/imaging-features'
    resample_dir = Path(__file__).parent.parent.joinpath('reference/standard_mesh_atlases/resample_fsaverage')
    
    orig_sphere_file = f'fsaverage_std_sphere.{HEMI}.{fsavg_res}_fsavg_{HEMI}.surf.gii'
    orig_sphere = os.path.join(resample_dir, orig_sphere_file)

    orig_area_file = f'fsaverage.{HEMI}.midthickness_va_avg.{fsavg_res}_fsavg_{HEMI}.shape.gii'
    orig_area   = os.path.join(resample_dir, orig_area_file)
    
    new_sphere_file = f'fs_LR-deformed_to-fsaverage.{HEMI}.sphere.{fslr_mesh}_fs_LR.surf.gii'
    new_sphere = os.path.join(resample_dir, new_sphere_file)

    new_area_file = f'fs_LR.{HEMI}.midthickness_va_avg.{fslr_mesh}_fs_LR.shape.gii'
    new_area   = os.path.join(resample_dir, 'fs_LR.{}.midthickness_va_avg.{}_fs_LR.shape.gii')

    # execute
    subprocess.call([
        'wb_command', 
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
    fsavg_res = fsavg_dict[fsavg_mesh]

    # parse hemisphere
    hemi, HEMI = parse_hemi(hemi)

    # parse whether we are resampling a label or 
    if label_or_metric == 'label':
        resample_type = '-label-resample'
    elif label_or_metric == 'metric':
        resample_type = '-metric-resample'
    else: 
        return

    # define relative paths to files
    resample_dir = Path(__file__).parent.parent.joinpath('reference/standard_mesh_atlases/resample_fsaverage')

    orig_sphere_file = f'fsaverage_std_sphere.{HEMI}.{fsavg_res}_fsavg_{HEMI}.surf.gii'
    orig_sphere = os.path.join(resample_dir, orig_sphere_file)

    orig_area_file = f'fsaverage.{HEMI}.midthickness_va_avg.{fsavg_res}_fsavg_{HEMI}.shape.gii'
    orig_area   = os.path.join(resample_dir, orig_area_file)
    
    new_sphere_file = f'fs_LR-deformed_to-fsaverage.{HEMI}.sphere.{fslr_mesh}_fs_LR.surf.gii'
    new_sphere = os.path.join(resample_dir, new_sphere_file)

    new_area_file = f'fs_LR.{HEMI}.midthickness_va_avg.{fslr_mesh}_fs_LR.shape.gii'
    new_area   = os.path.join(resample_dir, new_area_file)

    # execute
    subprocess.call([
        'wb_command', 
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


def split_dlabel(cifti_file, gii_left, gii_right):
    '''
    HCP workbench function call to split a dlabel cifti file into cortex only giftis

    Parameters
    ----------

    Returns
    ----------
    '''
    subprocess.call(['wb_command',
                '-cifti-separate',
                cifti_file, 
                'COLUMN',
                '-label', 'CORTEX_LEFT', gii_left, 
                '-label', 'CORTEX_RIGHT', gii_right])
    return gii_left, gii_right



