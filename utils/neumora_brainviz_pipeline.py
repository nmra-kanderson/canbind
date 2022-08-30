import os
import pathlib
import re
import traceback
import shutil
import glob
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List
import cairosvg
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nb
from mriqc.viz.utils import plot_mosaic
import click
import numpy as np
import tqdm

from utils.neuroimage import parse_session_hcp, get_qunex_dirs, print_feedback, valid_parcel_schema


@click.command()
@click.option('--scans', multiple=True)
@click.option('--sessions-dir', type=click.Path(exists=True))
@click.option('--output-dir')
@click.option('--regex-session-filter', default=None)
@click.option('--study-name', type=str)
@click.option('--n-processes', type=int, default=4)
def main(
        scans: List[str],
        sessions_dir: pathlib.PosixPath,
        output_dir: str,
        study_name: str,
        regex_session_filter: str,
        n_processes: int
    ):
    """
    Produce a set of PNG images from QuNex output.

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
    sessions_dir: pathlib.Path
        Path to the QuNex sessions folder
        Example: ~/fsx-mount/example/studies/embarc-20201122-LHzJPHi4/sessions
    output_dir: str
        Where to save the outputs of this script
        Example: ~/embarc/flat_files
    regex_session_filter: str
        Regex that provides a way of identifying what a session
        folder looks like.
        Example: '([0-9]{4})' to capture a BSNIP session like '0801'
        Example: '(.{6}_(baseline|wk1))' to capture EMBARC sessions like
            'MG0257_baseline' or 'CU0011_wk1'
    """


    # Instructions to be added to PNG images instructing the user on what to look for
    # This dictionary is non-standard/ugly, but keep it formatted this way. 
    # Otherwise the text doesn't display nicely in the PNG header
    # -----------------------------
    qc_instructions = {
    'subjT1w_mosaic': 
'Check quality of the anatomical scan\n\
Bad data will have Ringing, Ghosting, Poor GM/WM contrast',

    'subjMNI_BOLD': 
'Check BOLD-to-Anatomical alignment in group atlas space \n\
The Green/Purple outlines should be well aligned to boundaries of the brain',

    'subjMNI_Ribbon': 
'Check Freesurfer cortical ribbon in group atlas space\n\
The Green/Purple line should trace the outer surface of the brain',

    'atlasMNI_Ribbon': 
'Check accuracy of individual-to-group transform\n\
The Green/Purple line should *approximately* trace the outer boundary of the group atlas brain',

    'subjT1w_BrainMask': 
'Check brain extraction and brainmask\n\
The Yellow line should circle the subjects brain',

    'boldSignalIntensity':
'Check Mean BOLD signal intensity\n\
Check that the first parts of the timecourse are not outliers, \n\
otherwise we need to scrub the initial data points.'
    }
    scans = list(scans)
    print(f'Sessions Dir:          {sessions_dir}')
    print(f'Regex Session Filter:  {regex_session_filter}')
    print(f'Output Dir:            {output_dir}')
    print(f'Study Name:            {study_name}')
    print(f'Scans                : {scans}')
    print(f'N Processes:           {n_processes}')

    # TMP: manually set inputsa
    tmp = False
    if tmp == True:
        sessions_dir         = '/fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20220301-EuKEp5Gw/sessions'
        #sessions_dir         = '/fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-LHzJPHi4/sessions'
        regex_session_filter = '(.{6}_(baseline|wk1))'
        output_dir           = '/fmri-qunex/research/imaging/datasets/embarc/qunex_features'
        scans                = ['bold ert', 'bold rest run-1', 'bold rest run-2','bold reward']
        study_name = 'EMBARC'

    # Create output folder if it doesn't exist
    # --------------------
    study_output_dir = Path(output_dir, study_name)
    study_output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of QuNex subdirectories for processing
    session_dirs = get_qunex_dirs(sessions_dir, regex_session_filter)


    # Create PNG images for each scan in each session
    # Parallel plotting currently does not work. Image rendering interferes between sessions. 
    # --------------------
    #session_dir = session_dirs[3]
    #make_session_qc_pngs(session_dir)
    for i, session_dir in enumerate(session_dirs):
        try:
            print(f'{i+1}/{len(session_dirs)}: {session_dir}')
            make_session_qc_pngs(session_dir, study_output_dir, scans, qc_instructions)
        except Exception:
            print(traceback.format_exc())
            # TODO: Add actual error handling
            print('Visual QC Failed')


def make_session_qc_pngs(session_dir, study_output_dir, scans, qc_instructions):
    '''
    Make the QC images for a given QuNex session

    Parameters
    -----------
    session_dir: str
        Full path to a QuNex session output directory
            e.g. /fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-LHzJPHi4/sessions/UM0056_baseline
    '''

    # Create output directory for images
    raw_dir = Path(study_output_dir, 'raw', session_dir.stem, 'brainvis')
    raw_dir.mkdir(exist_ok=True)

    # read BOLD run information for this QuNex session
    bold_dict     = parse_session_hcp(Path(session_dir, 'session_hcp.txt'), scan_types=scans)
    bold_rev_dict = {x:y for x,y in zip(bold_dict.values(), bold_dict.keys())}

    # -------------------------------
    # Anatomical visualizations
    # -------------------------------
    anag_png_list = make_anat_visualizations(session_dir=session_dir, 
                                                output_folder=raw_dir, 
                                                ftype='T1w',
                                                qc_instructions=qc_instructions)

    # --------------------------------------
    # Signal Intensity Plots for BOLDs
    # --------------------------------------
    mni_results_dir  = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/Results')
    bold_list        = [Path(mni_results_dir, bold_num, f'{bold_num}.nii.gz') for bold_num in bold_dict.values()]
    bold_signal_plot = bold_mean_signal_plots(bold_list, 
                                              png_dir=raw_dir, 
                                              fname=f'{session_dir.stem}_bold_mean_signal_intensity.png', 
                                              qc_instructions=qc_instructions)

    # -------------------------------
    # BOLD/Functional visualizations
    # -------------------------------
    func_png_list = []
    for bold_num in bold_rev_dict.keys():
        bold_name = bold_rev_dict[bold_num].replace(' ', '_')
        png_path  = make_func_visualizations(session_dir=session_dir, 
                                            output_folder=raw_dir, 
                                            bold_num=bold_num, 
                                            bold_name=bold_name, 
                                            qc_instructions=qc_instructions)
        func_png_list.append(png_path)


    # -------------------------------
    # Copy QuNex QC PNGs
    # -------------------------------
    session_qc_dir = Path(sessions_dir, 'QC')
    bold_pngs   = [x for x in Path(session_qc_dir, 'BOLD').glob(f'*{session_dir.stem}*.png')]
    t1w_pngs    = [x for x in Path(session_qc_dir, 'T1w').glob(f'*{session_dir.stem}*.png')]
    mvmt_pngs   = [x for x in Path(session_qc_dir, 'movement/movement_plots/cor').glob(f'*{session_dir.stem}*.pdf')]
    dvars_pngs  = [x for x in Path(session_qc_dir, 'movement/movement_plots/dvars').glob(f'*{session_dir.stem}*.pdf')]
    dvarsme_pngs = [x for x in Path(session_qc_dir, 'movement/movement_plots/dvarsme').glob(f'*{session_dir.stem}*.pdf')]

    all_qunex_images = bold_pngs + t1w_pngs + mvmt_pngs + dvars_pngs + dvarsme_pngs
    for qunex_img in all_qunex_images:
        dest_img = Path(raw_dir, str(qunex_img).split('/')[-1])
        shutil.copy(qunex_img, dest_img)


def make_anat_visualizations(session_dir, output_folder, ftype='T1w', qc_instructions=None):
    '''
    Produce mosaic PNG visualizations of anatomical MRI data

    Parameters
    ----------
    session_dir: str
        Full path to a QuNex session output directory
            e.g. /fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-LHzJPHi4/sessions/UM0056_baseline
    output_folder: str
        Full path to the output directory
    ftype: str
        TODO: eventually this will allow us to plot T2w images as well as T1w
        Doesn't doe anything at the moment
    qc_instructions: dict
        Dictionary of descriptive/instructive text to add as a PNG header. 

    Returns
    ----------
    List of PNG filepaths
    '''
    # PNG output directory
    png_dir = Path(output_folder)

    # -----------------------------------
    # Check T1w alignment to MNI template
    # -----------------------------------
    # paths to the MRI nifti files to plot
    #mni_template = Path(__file__).parent.parent.joinpath('reference/mri_atlas/MNI152_T1_0.7mm.nii.gz')
    mni_template = '/home/ubuntu/Projects/qunex_utils/templates/MNI152_T1_1mm.nii.gz'
    mni_ribbon   = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/ribbon.nii.gz') 

    # make the plot
    fname = f'{session_dir.stem}_mni152_template_ribbon'
    out   = mosaic_xyz_wrapper(anat=mni_template, overlay=mni_ribbon, png_dir=png_dir, fname=fname)
    atlasMNI_Ribbon, f_underlay, f_overlay = out 
    # add header
    add_png_header_instructions(png_file=atlasMNI_Ribbon, 
                                message=qc_instructions['atlasMNI_Ribbon'], 
                                f_underlay=f_underlay, 
                                f_overlay=f_overlay)

    # -------------------------------------------------------
    # Check Freesurfer Ribbon alignment to subject T1w image
    # -------------------------------------------------------
    mni_t1 = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/T1w_restore.nii.gz') 
    # make the plot
    fname = f'{session_dir.stem}_subjT1wMNI_ribbon'
    out   = mosaic_xyz_wrapper(anat=mni_t1, 
                                overlay=mni_ribbon, 
                                png_dir=png_dir, 
                                fname=f'{session_dir.stem}_subjT1wMNI_ribbon')
    subjMNI_Ribbon, f_underlay, f_overlay = out  
    # add header
    add_png_header_instructions(png_file=subjMNI_Ribbon, 
                                message=qc_instructions['subjMNI_Ribbon'], 
                                f_underlay=f_underlay, 
                                f_overlay=f_overlay)

    # -----------------------
    # T1w Brain Mosaic
    # -----------------------
    # paths to niftis
    t1w_acpc = Path(session_dir, 'hcp', session_dir.stem, 'T1w/T1w_acpc.nii.gz') 
    save_svg = Path(output_folder, 't1w_acpc_mosaic.svg')

    # draw nifti mosaic 
    mosaic_png = Path(output_folder, f'{session_dir.stem}_t1w_acpc_mosaic.png')
    plot_mosaic(img=str(t1w_acpc),
                out_file=save_svg,
                zmax=30)
    # convert svg to png
    cairosvg.svg2png(url=str(save_svg), write_to=str(mosaic_png))
    os.remove(save_svg)
    # add header
    add_png_header_instructions(png_file=str(mosaic_png), 
                                message=qc_instructions['subjT1w_mosaic'], 
                                f_underlay=f_underlay)

    # -----------------------
    # T1w Brain Mask
    # -----------------------
    t1w_acpc        = Path(session_dir, 'hcp', session_dir.stem, 'T1w/T1w_acpc.nii.gz') 
    t1w_acpc_bmask  = Path(session_dir, 'hcp', session_dir.stem, 'T1w/T1w_acpc_brain_mask.nii.gz') 

    # make the plot
    out = mosaic_xyz_wrapper(anat=t1w_acpc, 
                             overlay=t1w_acpc_bmask, 
                             png_dir=png_dir, 
                             fname=f'{session_dir.stem}_subjT1w_BrainMask')
    subjT1w_BrainMask, f_underlay, f_overlay =  out
    # add header
    add_png_header_instructions(png_file=subjT1w_BrainMask, 
                                message=qc_instructions['subjT1w_BrainMask'], 
                                f_underlay=f_underlay, 
                                f_overlay=f_overlay)

    # list of the important pngs
    anat_pngs = [
        str(atlasMNI_Ribbon),
        str(subjMNI_Ribbon),
        str(mosaic_png),
        str(subjT1w_BrainMask)
    ]
    return anat_pngs


def plot_anat(anat, overlay, display_mode, png_dir=None, fname='anat'):
    '''
    Plot a volumetric nifti file with the NiLearn plotting library.
    Optionally include an overlay and save png to disk.

    Parameters
    ----------
    anat: str
        Full path to the anatomical nifti file (.nii or nii.gz) to be plotted as underlay.
    overlay: str
        Full path to a nifti file (.nii or nii.gz) to be plotted on top of the underlay
    display_mode: str
        Which dimension or plot type to produce
            - 'x': sagittal
            - 'y': coronal
            - 'z': axial
            - 'ortho': three cuts are performed in orthogonal
              directions
            - 'tiled': three cuts are performed and arranged
              in a 2x2 grid
            - 'mosaic': three cuts are performed along
              multiple rows and columns
    png_dir: str
        Full path to the output directory for writing PNG files
    fname: str
        Descriptive name for the PNG.

    Returns
    ----------
    display:  nilearn.plotting
    save_png: str
        Full path to the produced PNG file
    '''

    data_obj = np.array(nb.load(str(anat)).dataobj)
    vmin = np.percentile(data_obj[data_obj != 0], 2.5)
    vmax = np.percentile(data_obj[data_obj != 0], 97.5)

    display = plotting.plot_anat(anat_img=str(anat), 
                                display_mode=display_mode, 
                                draw_cross=False,
                                vmin=vmin, vmax=vmax)
    if overlay != None: 
        display.add_contours(img=str(overlay))

    # save 
    if png_dir != None: 
        save_png = Path(png_dir, f'{fname}_{display_mode}.png')
        #print(f'Saving PNG: {save_png}')
        display.savefig(save_png)
    return display, save_png                               


def make_func_visualizations(session_dir, output_folder, bold_num, bold_name, qc_instructions):
    '''
    Produce PNG visualizations for functional MRI/BOLD data

    Parameters
    ----------
    session_dir: str
        Full path to a QuNex session output directory
            e.g. /fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-LHzJPHi4/sessions/UM0056_baseline

    output_folder: str
        Full path to the output directory
    
    bold_num: str, int
        QuNex relabels BOLD images as numbers. 
        This parameter determines which BOLD run will be plotted.
    
    bold_name: str
        Descriptive name for the BOLD run,
        usually taken from the QuNex session_hcp.txt file
        e.g. "rest1"
        
    Returns
    ----------
    str/path
        Full path to the BOLD QC PNG
    '''
    png_dir = Path(output_folder)

    # plot MNI EPI volume on MNI x-formed anatomical 
    # --------------
    t1w_epi     = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear/Results', str(bold_num), '{}_SBRef.nii.gz'.format(str(bold_num)))
    t1w_base    = Path(session_dir, 'hcp', session_dir.stem, 'MNINonLinear', 'T1w_restore.nii.gz')

    # make the plot
    out = mosaic_xyz_wrapper(anat=t1w_base, 
                             overlay=t1w_epi, 
                             png_dir=png_dir, 
                             fname=f'{session_dir.stem}_subjMNI_BOLD_{bold_num}_{bold_name}')
    subjMNI_BOLD, f_underlay, f_overlay = out
    # add header
    add_png_header_instructions(png_file=subjMNI_BOLD, message=qc_instructions['subjMNI_BOLD'], f_underlay=f_underlay, f_overlay=f_overlay)

    return subjMNI_BOLD


def add_png_header_instructions(png_file, message, f_underlay=None, f_overlay=None):
    '''
    Add a descriptive text header to PNG images. 
    Tells the viewer what data is displayed and what visual features to look for. 

    Parameters
    ----------
    png_file: str
        Full path of the PNG file.
    
    message: str
        Text to write in the image header
        Can be a single line: 
            e.g. "Hello World"
        Or split across multiple lines with:
            e.g. "Hello\n World"

    f_underlay: str
        Filename for the brain data plotted as image underlay

    f_overlay: str
        Filename for the brain data plotted as image underlay
    '''
    # load brain imag
    brain_img = Image.open(str(png_file))

    # lines to write in header
    lines = message.split('\n')

    # add blank line
    if f_underlay != None or f_overlay != None:
        lines.append('\n')

    # height of the text PNG
    txt_height = 100
    if f_underlay != None:
        txt_height += 35
        lines.append('underlay:\t{}'.format(f_underlay))
    if f_overlay != None: 
        txt_height += 35
        lines.append('overlay:\t{}'.format(f_overlay))
        
    # create new text header png
    W, H     = (brain_img.width, txt_height)
    text_img = Image.new("RGBA",(W,H),"white")
    draw     = ImageDraw.Draw(text_img)
    myFont = ImageFont.truetype("Roboto-Regular.ttf", 22)
    
    # write each line separately (to allow for center-alignment)
    y_text = 10
    for line in lines: 
        w, h = draw.textsize(line, font=myFont)
        draw.text(((W-w)/2, y_text), line, fill="black", font=myFont)
        # the underlay/anat filenames are printed with a little vertical space below instructions
        if line == '\n':
            h=h-30
        y_text += h
    
    # vertically concatenate the text/brain-image PNGs
    combo_img = Image.new('RGB', (brain_img.width, brain_img.height + text_img.height))
    combo_img.paste(text_img, (0, 0))
    combo_img.paste(brain_img, (0, text_img.height))
    # save 
    combo_img.save(png_file)



def bold_mean_signal_plots(bold_list, png_dir, fname, qc_instructions):
    '''
    Plot the average intensity of BOLD fMRI data.
    This allows the viwer to identify major outliers at the beginning of a scan. 
    If mean signal in the first 10seconds is very high, it indiciates that the BOLD
    scane was not properly censored prior to analysis. 

    Parameters
    ----------
    bold_list: list
        List of full file paths to the BOLD fMRI nifti data
    
    png_dir: str
        Full path to the PNG output directory. 
    
    fname: str
        Descriptive filename for the PNG to write
    
    qc_instructions: dict
        Dictionary of descriptive/instructive text to add as a PNG header. 

    Returns
    ----------
    str
    '''
    
    fig, axs = plt.subplots(len(bold_list), figsize=(8, 6))
    plt.tight_layout()
    for i,check_nii in enumerate(bold_list):
        #print('Plotting Mean Signal: BOLD_{}'.format(i))
        nii_title   = '/'.join(str(check_nii).split('/')[-2:])
        nii_obj     = nb.load(check_nii)
        nii_dat     = nii_obj.get_fdata()
        mean_signal = np.mean(nii_dat, axis=(0,1,2))
        axs[i].plot(mean_signal)
        axs[i].title.set_text(nii_title)
    
    save_png = Path(png_dir, fname)
    plt.savefig(save_png)

    add_png_header_instructions(png_file=save_png, message=qc_instructions['boldSignalIntensity'])

    return save_png


def concat_pngs_vert(im1, im2, im3, png_dir, fname):
    '''
    Vertically concatenate 3 images

    Parameters
    ----------
    im1/im2/im3: PIL.PngImagePlugin.PngImageFile
        Loaded PIL image 
    
    png_dir: str
        Full path of PNG output directory
    
    fname: str
        Descriptive filename for the resulting PNG
    '''
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (0, im1.height + im2.height))

    out_path = Path(png_dir, fname)
    dst.save(out_path)

    return dst, out_path


def mosaic_xyz_wrapper(anat, overlay, png_dir, fname):
    '''
    Function to produce a mosaic brain plot in three view orientations (x/y/z)

    Parameters
    ----------
    anat: str
        Full path to the anatomical nifti file (.nii or nii.gz) to be plotted as underlay.
    overlay: str
        Full path to a nifti file (.nii or nii.gz) to be plotted on top of the underlay
    png_dir: str
        Full path to the PNG output directory
    fname: str
        Descriptive filename for the resulting PNG
    '''
    
    png_paths = []
    png_list  = []
    for display_mode in ['x','y','z']:
        mni_plot, png_path = plot_anat(anat=anat, 
                                        overlay=overlay, 
                                        display_mode=display_mode, 
                                        png_dir=png_dir, 
                                        fname=fname)
        png_paths.append(png_path)
    png_list = [Image.open(str(png_path)) for png_path in png_paths]
    
    # stich together the xyz PNGs into a single mosaic image
    out = concat_pngs_vert(png_list[0], 
                            png_list[1], 
                            png_list[2], 
                            png_dir=png_dir, 
                            fname=f'{fname}_xyz.png')
    combined_xyz_png, combined_xyz_png_path = out

    # remove individual pngs
    for png_path in png_paths:
        if os.path.exists(png_path):
            os.remove(png_path)
    f_underlay = '/'.join(str(anat).split('/')[-2:])
    f_overlay  = '/'.join(str(overlay).split('/')[-2:])

    return combined_xyz_png_path, f_underlay, f_overlay


if __name__ == '__main__':
    main()
