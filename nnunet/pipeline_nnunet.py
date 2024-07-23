#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Description:
nnU-Net pipeline for WMH segmentation, taking additionally care of:
    1. axes orientation - if not RAS+ or LAS+, image will be reoriented to RAS+
    2. image resolution - will be reported, but not adjusted, since nnU-Net is doing this
    3. image registration - optional; T1w will be registered to FLAIR using Slicer software; this step needs brain masks which will be created with HD-BET

'''

import sys, os, time, subprocess, argparse, re
from os.path import join, basename, dirname
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
from shutil import copy2, rmtree
from pathlib import Path
import string
import random
import pandas as pd
import torch

def run_subprocess(cmd, verbose, label):
    if verbose: print(f"Calling {label} command with:")
    if verbose: print(cmd, '\n')
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    # display output
    if output.returncode != 0:
        print("STDOUT/STDERR:")
        print(output.stdout.decode("utf-8"))
        raise ValueError(f"ERROR during call of {label} command! For stdout/stderr of the command see above!")
    else:
        if verbose: print(output.stdout.decode("utf-8"))

def qfrom_2_sform_and_compress(fname_in, fname_out=None):
    
    nii = nib.load(fname_in)
    nii.set_sform(nii.get_qform(), code="scanner")
    nii.set_qform(None, code="scanner")
    if nii.dataobj.slope==1 and nii.dataobj.inter==0:
        nii.header.set_slope_inter(1, 0)
    if fname_out is None:
        fname_out = fname_in
    if fname_out.endswith('.nii'):
        fname_remove = fname_out
        fname_out = re.sub('\.nii$', '.nii.gz', fname_out)
    nib.save(nii, fname_out)
    if 'fname_remove' in locals() and os.path.exists(fname_remove):
        Path(fname_remove).unlink()
    
    return  fname_out


def reorient_to_RAS(fname):
    nii = nib.load(fname)
    niiRAS = nib.as_closest_canonical(nii) 
    niiRAS.set_qform(niiRAS.get_sform(),code='aligned') #-- required, because 'as_closest_canonical' deletes the qform
    if nii.dataobj.slope==1 and nii.dataobj.inter==0:
        niiRAS.header.set_slope_inter(1, 0)
    nib.save(niiRAS, fname)

def hdbet(head, brain, verbose=True):
    if not torch.cuda.is_available():
        cmd = f'hd-bet -i "{head}" -o "{brain}" -device cpu -mode fast'
    else:
        cmd = f'hd-bet -i "{head}" -o "{brain}"'
    run_subprocess(cmd, verbose, label="HDBET")
    

def dilate_brainmask(fn, fnNew=None):
    nii = nib.load(fn)
    arr = nii.get_fdata()
    # Dilate image with connectivity 3 and 2 iterations
    struct = ndimage.generate_binary_structure(arr.ndim, 3)
    arr = ndimage.binary_dilation(arr!=0, structure=struct, iterations=2)
    nii = nib.Nifti1Image(arr, nii.affine, nii.header)
    # Copy qform to sform, to ensure consistent headers in all input images
    nii.set_sform(nii.get_qform(),code='scanner')
    nii.set_qform(None,code='scanner')
    nii.set_data_dtype('uint8')
    if fnNew is None:
        fnNew = fn
    nib.save(nii, fnNew)

def register_with_slicer(fixed, moving, fixedMask, movingMask, registered, verbose=True):
    registered = re.sub('\.nii(\.gz)?$', '', registered) #- remove extension, which will be added below
    slicer = (f'/opt/BRAINSFit/BRAINSFit --fixedVolume "{fixed}" --movingVolume "{moving}" --initializeTransformMode useCenterOfROIAlign'
    f' --outputTransform "{registered}.h5" --outputVolume "{registered}.nii.gz"' +
    f' --outputVolumePixelType float --samplingPercentage 0.1' +
    f' --transformType Rigid,Affine --translationScale 700 --interpolationMode WindowedSinc' +
    f' --maskProcessingMode ROI --fixedBinaryVolume "{fixedMask}" --movingBinaryVolume "{movingMask}"' +
    f' --echo')
    
    run_subprocess(slicer, verbose, label="SLICER's BRAINSFit")


def resample(image, reference=None, resampled=None, interpolator=None, transform=None):
    # Output image Origin, Spacing, Size, Direction are taken from the reference image
    if reference is None:
        reference = image
    if resampled is None:
        resampled = image
    if transform is None:
        # print('Using identity transform')
        dimension = 3
        transform = sitk.Transform(dimension, sitk.sitkIdentity)
    if interpolator is None:
        # print('Using nearest neighbor interpolation')
        interpolator=sitk.sitkNearestNeighbor
    default_value = 0

    img = sitk.ReadImage(image)
    ref = sitk.ReadImage(reference)

    res = sitk.Resample(img, ref, transform,
                         interpolator, default_value)
    
    sitk.WriteImage(res, resampled)
    

def extract_statistics(fn):
    nii = nib.load(fn)
    arr = nii.get_fdata()
    voxels = np.count_nonzero(arr)
    volume = voxels * np.prod(nii.header.get_zooms()[0:3])
    struct = ndimage.generate_binary_structure(arr.ndim, connectivity=1)
    _, clusters = ndimage.label(arr, struct)
    df = pd.DataFrame({'Name': ['volume','voxels','clusters'], 'Value': [round(volume,1), voxels, clusters], 'Unit': ['mm^3','count','count']}, dtype='object')
    return df

def nnunet_prediction(dirInput, verbose=True):
    cmd = (
        'nnUNet_predict'
        f' -i "{dirInput}" -o "{dirInput}"'
        ' -tr nnUNetTrainerV2 -m 3d_fullres -p nnUNetPlansv2.1_8GB_iso1mm -t Task700_WMH'
    )
    run_subprocess(cmd, verbose, label="nnU-Net")


def pipeline_nnunet(flair, t1, wmh_mask, skipRegistration=False, saveStatistics=False, verbose=True, debug=False):
    
    start_script = time.time()
    if verbose: print(f"Segmenting WMH from:\n"
        f"  {flair}\n"
        f"  {t1}")
    if verbose: print(f"Output label map will be written to:\n"
        f"  {wmh_mask}\n")
    
    if os.path.exists(wmh_mask) and verbose:
        print('Warning: output label map exists already and will be overwritten\n')

    strRand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    dirTemp = re.sub('\.nii(\.gz)?$', '_temp-'+strRand, wmh_mask)
    if verbose: print(f"Creating temporary folder for processing:\n"
                      f"  {dirTemp}")
    if os.path.exists(dirTemp):
        print('Warning: temporary folder exists already and will be removed')
        rmtree(dirTemp)
    Path(dirTemp).mkdir(parents=True)

    # Copy FLAIR and T1w image to a temporary folder
    flair_in = flair
    flair = join(dirTemp, basename(flair))
    copy2(src=flair_in, dst=flair)
    t1_in = t1
    t1 = join(dirTemp, basename(t1))
    copy2(src=t1_in, dst=t1)
   
    # Copy qform to sform, to ensure consistent header
    flair = qfrom_2_sform_and_compress(flair)
    t1 = qfrom_2_sform_and_compress(t1)

    # Check and fix input images if necessary    
    if verbose: print('\nChecking input images:')
    niiFL = nib.load(flair)
    niiT1 = nib.load(t1)
    
    # Check and correct orientation
    axcodesFL = nib.aff2axcodes(niiFL.affine)
    axcodesT1 = nib.aff2axcodes(niiT1.affine)
    if verbose: print(f'FLAIR axes orientation is {"".join(axcodesFL)}+')
    if verbose: print(f'T1w   axes orientation is {"".join(axcodesT1)}+')
    if skipRegistration and axcodesT1 != axcodesFL:
            raise ValueError('When providing registered FLAIR and T1w as input, the axes orientation must match! Please check registration of input images or let the pipeline do the registration!')
    reorient_flag = False
    if axcodesFL != ('R','A','S') and axcodesFL != ('L','A','S'):
        if verbose: print('Reorienting FLAIR to RAS+')
        reorient_to_RAS(flair)
        reorient_flag = True
    if reorient_flag and skipRegistration:
        if verbose: print('Reorienting T1w to RAS+')
        reorient_to_RAS(t1)
    if verbose: print('')

    # Check resolution (no action required, since resolution is handled by nnU-Net)
    zoomsFL = np.array(niiFL.header.get_zooms()[0:3])
    if verbose: print(f'FLAIR resolution is {zoomsFL}')
    zoomsT1 = np.array(niiT1.header.get_zooms()[0:3])
    if verbose: print(f'T1w   resolution is {zoomsT1}')
    if skipRegistration and (zoomsT1 != zoomsFL).any():
            raise ValueError('When providing registered FLAIR and T1w as input, the resolution must match! Please check registration of input images or let the pipeline do the registration!')
    if verbose: print('')


    if skipRegistration:

        t1_rFlair = t1
    
    else:

        # Create brain mask for FLAIR
        flair_brain = re.sub('\.nii(\.gz)?$', '_brain.nii.gz', flair)
        flair_bmask = re.sub('\.nii(\.gz)?$', '_brain_mask.nii.gz', flair)
        start = time.time()
        hdbet(flair, flair_brain, verbose)
        end = time.time()
        if verbose: print(f"Duration HDBET: {end - start}\n")
        # Create brain mask for T1w
        t1_brain = re.sub('\.nii(\.gz)?$', '_brain.nii.gz', t1)
        t1_bmask = re.sub('\.nii(\.gz)?$', '_brain_mask.nii.gz', t1)
        start = time.time()
        hdbet(t1, t1_brain, verbose)
        end = time.time()
        if verbose: print(f"Duration HDBET: {end - start}\n")
        
        # Dilate brain masks
        flair_bmask_dil = re.sub('\.nii(\.gz)?$', '_dilated-c3-i2.nii.gz', flair_bmask)
        dilate_brainmask(flair_bmask, flair_bmask_dil)
        t1_bmask_dil = re.sub('\.nii(\.gz)?$', '_dilated-c3-i2.nii.gz', t1_bmask)
        dilate_brainmask(t1_bmask, t1_bmask_dil)

        # Run registration
        start = time.time()
        t1_rFlair = re.sub('\.nii(\.gz)?$', '_rFLAIR.nii.gz', t1)
        register_with_slicer(flair, t1, flair_bmask_dil, t1_bmask_dil, t1_rFlair, verbose)
        end = time.time()
        if verbose: print(f"Duration registration: {end - start}\n")

    
    # Copy FLAIR and registered-T1 to dedicated nnU-Net folder, adding suffix "_0000/_0001" as modality identifier for nnU-Net
    dirNnunet = join(dirTemp, 'nnUNet')
    Path(dirNnunet).mkdir()
    if verbose: print('Organising input for nnU-Net:')
    temp = copy2(src=flair,     dst=join(dirNnunet, 'wmh_0000.nii.gz'))
    if verbose: print(f'   {flair} -> {temp}')
    temp = copy2(src=t1_rFlair, dst=join(dirNnunet, 'wmh_0001.nii.gz'))
    if verbose: print(f'   {t1_rFlair} -> {temp}\n')

    # Predict WMH using nnU-Net
    start = time.time()
    labelmap = nnunet_prediction(dirNnunet, verbose)
    end = time.time()
    if verbose: print(f"Duration nnU-Net: {end - start}\n")
    
    # Expected output filename is constructed by removing modality index
    labelmap = join(dirNnunet, 'wmh.nii.gz')
    qfrom_2_sform_and_compress(labelmap)

    if reorient_flag:
        if verbose: print('\nReorienting label map according to original axes orientation')    
        resample(image=labelmap, reference=flair_in, resampled=wmh_mask, interpolator=sitk.sitkNearestNeighbor)
        if verbose: print(f'Label map was saved to: {wmh_mask}')

    else:
        if verbose: print('\nNo postprocessing needed')
        copy2(labelmap, wmh_mask)
        if verbose: print(f'Label map was saved to: {wmh_mask}')
    
    # Extract statistics
    if verbose or saveStatistics:
        df = extract_statistics(wmh_mask)
        if verbose: print(f'\nStatistics for WMH lesions in output mask:\n ', df)
        if saveStatistics:
            wmh_stats = re.sub('\.nii(\.gz)?$', '.csv', wmh_mask)
            df.to_csv(wmh_stats, index=False)



    # Clean up
    if debug:
        if verbose: print(f'\nKeeping temporary folder: {dirTemp}')
    else:
        if verbose: print(f'\nDeleting temporary folder: {dirTemp}')
        rmtree(dirTemp)
    end = time.time()
    if verbose: print(f"\nTotal duration: {end - start_script}\n")



def isNIfTI(s):
    if os.path.isfile(s) and (s.endswith('.nii.gz') or s.endswith('.nii')):
        return s
    elif os.path.isfile(s+'.nii.gz'):
        return s+'.nii.gz'
    elif os.path.isfile(s+'.nii'):
        return s+'.nii'
    else:
        raise argparse.ArgumentTypeError("File path does not exist or is not NIfTI. Please check: %s"%(s))
    
def isSuffix(s):
    if len(re.sub('\.nii(\.gz)?$', '', s)) > 0:
        return re.sub('\.nii(\.gz)?$', '', s)
    else:
        raise argparse.ArgumentTypeError("String is not suited as suffix. Please check: %s"%(s))

class CustomArgumentParser(argparse.ArgumentParser):
    # This subclass ensures that single dash options have to be one character long (after the dash) and separated from their arguments by a space
    def parse_args(self, args=None, namespace=None):
        for arg_string in args:
            if arg_string.startswith('-') and not arg_string.startswith('--'):
                if len(arg_string) > 2 and not arg_string[2].isspace():
                    raise ValueError(f'Single dash options have to be one character long (after the dash) and separated from their arguments by a space. Argument "{arg_string}" violates this requirement.')
        args = super().parse_args(args, namespace)
        return args
    
def iniParser():
    parser = CustomArgumentParser(description="Predict WMH from FLAIR and T1w image. Resulting WMH mask will be in FLAIR space.")
    group0 = parser.add_argument_group('required arguments')
    group0.add_argument("--flair", dest="fnFLAIR", type=isNIfTI, help="path to input FLAIR image in NIfTI format")
    group0.add_argument("--t1", dest="fnT1", type=isNIfTI, help="path to input T1w image in NIfTI format")
    group1 = parser.add_argument_group('optional arguments')
    group1.add_argument("-s", "--suffix", type=isSuffix, default='_wmh', help="suffix appended to FLAIR file path (before extension), in order to create path to which to write WMH mask (defaults to '_wmh')")
    group1.add_argument("-o", "--fnOut", type=str, help="path to which to write the WMH mask as NIfTI file (overrides option '-s')")
    group1.add_argument("-d", "--dirOut", type=str, help="path to output folder, to which to write the WMH mask (if missing the parent folder of the path provided either with option '-o' or with option '--flair' will be used)")
    group1.add_argument("-x", "--overwrite", action='store_true', help="allow overwriting output file if existing. By default, already existing output will raise an error.")
    group1.add_argument("--saveStatistics", action='store_true', help="save WMH lesion statistics (i.e. volume, voxels, clusters) to CSV file. Filename will be the same as for the output WMH mask, but with extension '.csv'")
    group1.add_argument("--skipRegistration", action='store_true', help="skip affine registration of T1 to FLAIR, assuming that these images are already in register.")
    group1.add_argument("--debug", action='store_true', help="don't delete temporary processing folder to allow debugging.")
    group1.add_argument("--quiet", dest="quiet", action='store_true', help="suppress standard output to command line. Errors will still be displayed.")
    return parser

if __name__ == "__main__":

    parser = iniParser()
    if len(sys.argv)<2:
        parser.print_help()
        sys.exit(0)
    else:
        args = parser.parse_args(sys.argv[1::])
    
    verbose = not(args.quiet)
    if verbose:
        print("Running: " + " ".join([basename(sys.argv[0])]+sys.argv[1::]))

    # check essential input
    if not args.fnFLAIR:
        raise ValueError('Please provide FLAIR image as input, using option "--flair"')
    if not args.fnT1:
        raise ValueError('Please provide T1w image as input, using option "--t1"')
    
    # build output filename
    if args.fnOut:
        assert args.fnOut.endswith('.nii.gz'), f'Provided output file extension has to be ".nii.gz". Please check: {args.fnOut}'
    else:
        args.fnOut = re.sub('\.nii(\.gz)?$', '', args.fnFLAIR) + args.suffix + '.nii.gz'
    if args.dirOut:
        args.fnOut = join(args.dirOut, basename(args.fnOut))

    # check availability GPU
    if not torch.cuda.is_available():
        if verbose: print('\nWarning: No GPU found! HD-BET and nnU-Net will run on CPU, which could lead to slightly differnt results!\n')

    # check whether output exists already, and raise error, if overwrite is false
    if os.path.exists(args.fnOut):
        if not args.overwrite:
            raise ValueError(f'Output label map "{args.fnOut}" exists already. If you want to overwrite, use option "-x".')

    pipeline_nnunet(args.fnFLAIR, args.fnT1, args.fnOut, args.skipRegistration, args.saveStatistics, verbose, args.debug)
    
