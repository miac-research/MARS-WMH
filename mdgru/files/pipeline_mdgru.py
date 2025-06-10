#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Description:
MD-GRU pipeline for WMH segmentation, taking additionally care of:
    1. axes orientation - if not RAS+ or LAS+, image will be reoriented to RAS+
    2. image resolution - has to be within range [0.95 mm, 1.05 mm] in each dimension, 
        otherwise will be resliced to 1 mm isotropic
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
from create_qc_image import create_qc_image
from create_html_with_png import create_html_with_png


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


def nifti_sanity_check(fname, label):
    
    nii = nib.load(fname)
    sform = nii.header.get_sform()
    qform = nii.header.get_qform()
    if nib.aff2axcodes(sform) != nib.aff2axcodes(qform):
        print(f"\nWarning: 'sform' and 'qform' of affine matrix in {label} image have different axes orientation!")
        
        if np.all(nii.header.get_best_affine() == qform):
            print("Warning: Using qform, as suggested by Nibabel's 'best_affine'")
            nii.set_sform(qform, code="scanner")
            nib.save(nii, fname)
        elif np.all(nii.header.get_best_affine() == sform):
            print("Warning: Using sform, as suggested by Nibabel's 'best_affine'")
            nii.set_qform(sform, code="scanner") #--- remember that qform cannot store sheers!
            nib.save(nii, fname)
        else:
            print("Warning: It is not possible to figure out, which one is correct! Trying with qform!")
            #--- Nothing to do here, since this is done anyways later

        print("Warning: However, please check resulting registration of T1w to FLAIR!!!")


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

def apply_mask(fname_image, fname_mask, fname_out=None):
    img = sitk.ReadImage(fname_image)
    mask = sitk.ReadImage(fname_mask)
    arr = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(mask)
    arr[mask<=0] = 0
    res = sitk.GetImageFromArray(arr)
    res.CopyInformation(img)
    if fname_out is None:
        fname_out = fname_image
    sitk.WriteImage(res, fname_out)

def change_spacing(image, resampled, out_spacing=[1.0, 1.0, 1.0], interpolator=None):
    
    # Resample images to 1mm isotropic spacing

    if resampled is None:
        resampled = image
    
    img = sitk.ReadImage(image)

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(img.GetPixelIDValue())

    if interpolator is None:
        # print('Using nearest neighbor interpolation')
        interpolator=sitk.sitkNearestNeighbor
    resample.SetInterpolator(interpolator)

    resT = resample.Execute(img)

    # Set negative values to zero
    arr = sitk.GetArrayFromImage(resT)
    arr[arr<0] = 0
    res = sitk.GetImageFromArray(arr)
    res.CopyInformation(resT)

    sitk.WriteImage(res, resampled)

def hdbet(head, brain, verbose=True):
    
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        arch_str = f"sm_{major}"
        if any(string.startswith(arch_str) for string in torch.cuda.get_arch_list()):
            gpu_available = True
        else:
            print(f'CUDA capability sm_{major}{minor} is not compatible with the current PyTorch installation.\nUsing CPU instead of GPU for HD-BET.')
            gpu_available= False  
    else:
        gpu_available = False

    if gpu_available:
        cmd = f'hd-bet -i "{head}" -o "{brain}"'
    else:
        cmd = f'hd-bet -i "{head}" -o "{brain}" -device cpu -mode fast -tta 0'
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

def flip_axes(fnIn, fnOut=None, axis=0):
    # Load image and flip one or multiple axes
    img = sitk.ReadImage(fnIn)
    arr = sitk.GetArrayFromImage(img)
    # With sitk axes order is reversed with respect to Nibabel, therefore start counting with last axis
    if isinstance(axis, (list, tuple)):
        axis = tuple([-1 - x for x in axis])
    elif axis is not None:
        axis = -1 - axis
    arr = np.flip(arr, axis)
    res = sitk.GetImageFromArray(arr)
    res.CopyInformation(img)
    if fnOut is None: fnOut = fnIn
    sitk.WriteImage(res, fnOut)
    
def extract_statistics(fn):
    nii = nib.load(fn)
    arr = nii.get_fdata()
    voxels = np.count_nonzero(arr)
    volume = voxels * np.prod(nii.header.get_zooms()[0:3])
    struct = ndimage.generate_binary_structure(arr.ndim, connectivity=1)
    _, clusters = ndimage.label(arr, struct)
    df = pd.DataFrame({'Name': ['volume','voxels','clusters'], 'Value': [round(volume,1), voxels, clusters], 'Unit': ['mm^3','count','count']}, dtype='object')
    return df

def mdgru_prediction(flair, t1, fnOutTrunk='mdgru', verbose=True):

    dir_input = dirname(flair)
    modelCkpt="/model/WMH_170000"
    cmd = (
        'python3 /opt/mdgru/RUN_mdgru.py'
        f' -f "{basename(flair)}" "{basename(t1)}" --optionname "{fnOutTrunk}"'
        f' --datapath "{dir_input}"'  
        f' --locationtesting . --locationtraining .  --locationvalidation .'
        f' --ckpt "{modelCkpt}"'
        f' --num_threads 4 -w 100 100 100 -p 20 20 20'
        f' --dont_correct_orientation  --nclasses 2 --only_test'
    )
    run_subprocess(cmd, verbose, label="MD-GRU")

    # expected output filenames
    labelmap = join(dir_input, fnOutTrunk+"-labels.nii.gz")
    probdist = join(dir_input, fnOutTrunk+"-probdist.nii.gz")
    
    return probdist, labelmap


def pipeline_mdgru(flair, t1, wmh_mask, skipRegistration=False, flip=False, saveStatistics=False, verbose=True, debug=False):
    
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

    # Sanity check of NIfTI header
    nifti_sanity_check(flair, 'FLAIR')
    nifti_sanity_check(t1, 'T1w')
   
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

    # Check and correct image resolution 
    zoomsFL = np.array(niiFL.header.get_zooms()[0:3])
    if verbose: print(f'FLAIR resolution is {zoomsFL}')
    zoomsT1 = np.array(niiT1.header.get_zooms()[0:3])
    if verbose: print(f'T1w   resolution is {zoomsT1}')
    if skipRegistration and (zoomsT1 != zoomsFL).any():
            raise ValueError('When providing registered FLAIR and T1w as input, the resolution must match! Please check registration of input images or let the pipeline do the registration!')
    reslice_flag = False
    if any(zoomsFL < 0.95) or any(zoomsFL > 1.05) :
        if verbose: print('FLAIR resolution is out of expected range [0.95,1.05]')
        if verbose: print('Reslicing FLAIR into 1 mm isotropic voxel grid')
        # interpolator = sitk.sitkHammingWindowedSinc
        interpolator = sitk.sitkBSpline
        change_spacing(flair, flair, out_spacing=[1.0, 1.0, 1.0], interpolator=interpolator)
        reslice_flag = True
    if reslice_flag and skipRegistration:
        if verbose: print('Reslicing T1w into 1 mm isotropic voxel grid')
        # interpolator = sitk.sitkHammingWindowedSinc
        interpolator = sitk.sitkBSpline
        change_spacing(t1, t1, out_spacing=[1.0, 1.0, 1.0], interpolator=interpolator)
    if verbose: print('')


    # Create brain mask for FLAIR
    flair_brain = re.sub('\.nii(\.gz)?$', '_brain.nii.gz', flair)
    flair_bmask = re.sub('\.nii(\.gz)?$', '_brain_mask.nii.gz', flair)
    start = time.time()
    hdbet(flair, flair_brain, verbose)
    end = time.time()
    if verbose: print(f"Duration HDBET: {end - start}\n")

    if skipRegistration:
        t1_rFlair = t1
    else:
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

    
    if verbose: print('Applying brain mask to FLAIR and T1w')
    flair_noBet = re.sub('\.nii(\.gz)?$', '_noBet.nii.gz', flair)
    t1_rFlair_noBet = re.sub('\.nii(\.gz)?$', '_noBet.nii.gz', t1_rFlair)
    copy2(flair, flair_noBet)
    copy2(t1_rFlair, t1_rFlair_noBet)
    apply_mask(flair, flair_bmask)
    apply_mask(t1_rFlair, flair_bmask)
    if verbose: print('')

    # Predict WMH using MD-GRU
    start = time.time()
    probdist, labelmap = mdgru_prediction(flair, t1_rFlair, 'mdgru', verbose)
    end = time.time()
    if verbose: print(f"Duration MD-GRU: {end - start}\n")   

    # Reinsert sform, because mdgru deletes the sform
    qfrom_2_sform_and_compress(probdist)
    qfrom_2_sform_and_compress(labelmap)

    # Keep only the probability map for label 1 (label 0 is the background and is complementary)
    nii = nib.load(probdist)
    nii = nii.slicer[..., 1]
    nii.to_filename(probdist)

    # Predict WMH for the flipped images
    if not flip:
        probdist_avg = probdist
        labelmap_avg = labelmap

    else:
        # flip images
        if verbose: print('Flipping images left/richt and running MD-GRU again:')
        flair_flipped = re.sub('\.nii(\.gz)?$', '_flipped.nii.gz', flair)
        flip_axes(flair, flair_flipped, axis=0)
        t1_rFlair_flipped = re.sub('\.nii(\.gz)?$', '_flipped.nii.gz', t1_rFlair)
        flip_axes(t1_rFlair, t1_rFlair_flipped, axis=0)
        if verbose: print('')
        # Predict WMH for flipped images
        start = time.time()
        probdistFlipped, labelmapFlipped = mdgru_prediction(flair_flipped, t1_rFlair_flipped, 'mdgru-for-flipped', verbose)
        end = time.time()
        if verbose: print(f"Duration MD-GRU: {end - start}\n")

        # Reinsert sform, because mdgru deletes the sform
        qfrom_2_sform_and_compress(probdistFlipped)
        qfrom_2_sform_and_compress(labelmapFlipped)

        # Keep only the probability map for label 1 (label 0 is the background and is complementary)
        nii = nib.load(probdistFlipped)
        nii = nii.slicer[..., 1]
        nii.to_filename(probdistFlipped)

        # Reverse L/R flip
        if verbose: print('Flipping MD-GRU output back to original left/right orientation')
        flip_axes(probdistFlipped, probdistFlipped, axis=0)
        flip_axes(labelmapFlipped, labelmapFlipped, axis=0)

        # Average the probability maps for flipped and none flipped images
        if verbose: print('Averaging the probability maps for flipped and non-flipped images')
        img1 = sitk.ReadImage(probdist)
        img2 = sitk.ReadImage(probdistFlipped)
        arr1 = sitk.GetArrayFromImage(img1)
        arr2 = sitk.GetArrayFromImage(img2)
        arr = np.mean( np.stack((arr1,arr2)) , axis=0 )
        res = sitk.GetImageFromArray(arr)
        res.CopyInformation(img1)
        probdist_avg = re.sub('\.nii(\.gz)?$', '_averaged.nii.gz', probdist)
        sitk.WriteImage(res, probdist_avg)
        
        # Threshold the averaged probability map, to get a new labelmap (although this is not always used later)
        if verbose: print('Thresholding the averaged probability map')
        arr = (arr >= 0.5) * 1
        res = sitk.GetImageFromArray(arr)
        res.CopyInformation(img1)
        labelmap_avg = re.sub('\.nii(\.gz)?$', '_averaged.nii.gz', labelmap)
        sitk.WriteImage(res, labelmap_avg)

    apply_mask(labelmap_avg, flair_bmask)

    if reslice_flag:
        '''
        Reslicing reverses both:
            1. reorientation to RAS+ (if this was done at the beginning)
            2. reslicing into 1 mm  isotropic space (if this was done at the beginning)
        '''
        if verbose: print('Applying brain mask')
        apply_mask(probdist_avg, flair_bmask)
        if verbose: print('Reslicing probability map into original voxel grid and thresholding it')
        probdistResampled = probdist_avg.replace('.nii.gz',f'_resampled-as-input.nii.gz')
        # interpolator = sitk.sitkHammingWindowedSinc
        interpolator = sitk.sitkBSpline
        start = time.time()
        resample(image=probdist_avg, reference=flair_in, resampled=probdistResampled, interpolator=interpolator)
        nii = nib.load(probdistResampled)
        arr = nii.get_fdata()
        arr = (arr >= 0.5) * 1
        end = time.time()
        if verbose: print(f"Duration reslicing: {end - start}\n")
        if verbose: print(f'Saving label map to: {wmh_mask}')
        nii = nib.Nifti1Image(arr, nii.affine, nii.header)
        nii.set_data_dtype("uint8")
        nii.header.set_slope_inter(1, 0)
        nib.save(nii, wmh_mask)

    elif reorient_flag:
        if verbose: print('\nApplying brain mask to segmentation')
        if verbose: print('Reorienting label map according to original axes orientation')
        labelmapResampled = labelmap_avg.replace('.nii.gz',f'_resampled-as-input.nii.gz')
        resample(image=labelmap_avg, reference=flair_in, resampled=labelmapResampled, interpolator=sitk.sitkNearestNeighbor)
        if verbose: print(f'Saving label map to: {wmh_mask}')
        copy2(labelmapResampled, wmh_mask)

    else:
        if verbose: print('\nApplying brain mask to segmentation')
        if verbose: print(f'Saving label map to: {wmh_mask}')
        copy2(labelmap_avg, wmh_mask)

    
    # Extract statistics
    if reslice_flag and verbose:
        df = extract_statistics(labelmap_avg)
        print(f'\nStatistics for WMH lesions in the 1 mm isotropic resliced mask:\n', df)
    df = extract_statistics(wmh_mask)
    if verbose: print(f'\nStatistics for WMH lesions in output mask:\n ', df)
    if saveStatistics:
        wmh_stats = re.sub('\.nii(\.gz)?$', '.csv', wmh_mask)
        df.to_csv(wmh_stats, index=False)

    # Create QC
    if not args.omitQC:
        fnHTML = re.sub('\.nii(\.gz)?$', '', wmh_mask) + '_QC.html'
        if verbose: print(f'\nSaving QC image to:\n ', fnHTML)
        fnPNG = create_qc_image([flair_noBet, t1_rFlair_noBet], labelmap_avg, flair_bmask)
        text=None
        if reslice_flag: text = ['Please note:',
                                 'Showing resliced images (1 mm isotropic voxels), given to MD-GRU as input.',
                                 'However, the WMH mask is returned in the space of the original FLAIR, and WMH volume and voxel/cluster counts are calculated afterwards.']
        create_html_with_png(fnPNG, fnHTML, df, text)


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
    group0.add_argument("--flair", required=True, metavar='NIfTI', dest="fnFLAIR", type=isNIfTI, help="path to input FLAIR image in NIfTI format")
    group0.add_argument("--t1", required=True, metavar='NIfTI', dest="fnT1", type=isNIfTI, help="path to input T1w image in NIfTI format")
    group1 = parser.add_argument_group('optional arguments')
    group1.add_argument("-s", "--suffix", type=isSuffix, default='_wmh', help="suffix appended to FLAIR file path (before extension), in order to create path to which to write WMH mask (defaults to '_wmh')")
    group1.add_argument("-o", "--fnOut", type=str, help="path to which to write the WMH mask as NIfTI file (overrides option '-s')")
    group1.add_argument("-d", "--dirOut", type=str, help="path to output folder, to which to write the WMH mask (if missing the parent folder of the path provided either with option '-o' or with option '--flair' will be used)")
    group1.add_argument("-x", "--overwrite", action='store_true', help="allow overwriting output file if existing. By default, already existing output will raise an error.")
    group1.add_argument("--saveStatistics", action='store_true', help="save WMH lesion statistics (i.e. volume, voxels, clusters) to CSV file. Filename will be the same as for the output WMH mask, but with extension '.csv'")
    group1.add_argument("--omitQC", action='store_true', help="omit QC. By default an HTML will be created (names as the WMH mask, with suffix '_QC.html') showing FLAIR and T1w with WMH (red) and brain mask (cyan contour) as overlay.")
    group1.add_argument("--skipRegistration", action='store_true', help="skip affine registration of T1 to FLAIR, assuming that these images are already in register.")
    group1.add_argument("--flip", action='store_true', help="run MD-GRU twice, once on the unmodified images and once on the left/right flipped images.")
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
        if verbose: print('\nWarning: No GPU found! HD-BET and MD-GRU will run on CPU, which could lead to slightly differnt results!\n')

    # check whether output exists already, and raise error, if overwrite is false
    if os.path.exists(args.fnOut):
        if not args.overwrite:
            raise ValueError(f'Output label map "{args.fnOut}" exists already. If you want to overwrite, use option "-x".')

    pipeline_mdgru(args.fnFLAIR, args.fnT1, args.fnOut, args.skipRegistration, args.flip, args.saveStatistics, verbose, args.debug)
        
