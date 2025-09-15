#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
import os, string, random, re
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache_'+''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import nibabel as nib
import imageio.v3 as iio

#
def crop_nonzero(images, mask, border, thr=0, onlyZ=False, verbose = True):
    
    #- find ranges containing mask
    nz =np.nonzero(mask > thr)
    if not all([len(x) for x in nz]):
        print('Warning: Mask is empty!')
        print('Returning size of image')
        idxLow  = np.asarray([0,0,0])
        idxHigh = np.asarray(mask.shape)
    else:
        idxLow  = np.asarray([min(nz[0]),min(nz[1]),min(nz[2])])
        idxHigh = np.asarray([max(nz[0]),max(nz[1]),max(nz[2])])
        # ranges = idxHigh - idxLow +1
    
    if verbose: print("index low: ", idxLow)
    if verbose: print("index high:", idxHigh)

    # add border, if possible
    if border is not None:
        if verbose: print(f"Adding border of {border} voxels")
        idxLow  = np.maximum(idxLow-border, [0,0,0])
        idxHigh = np.minimum(idxHigh+border, np.array(img.shape[0:2])-1)
        if verbose: print("index low: ", idxLow)
        if verbose: print("index high:", idxHigh)
    
    ranges = idxHigh - idxLow + 1
    # print('Ranges:', ranges)
    
    if ranges[2]<6:
        idxLow[2]  = np.maximum(idxLow[2]-np.floor((6-ranges[2])/2), 0)
        idxHigh[2] = np.minimum(idxHigh[2]+np.ceil((6-ranges[2])/2), np.array(mask.shape[2])-1)

    # crop images
    for iImg, img in enumerate(images):
        if onlyZ:
            images[iImg] = img[:,:, idxLow[2]:idxHigh[2]+1]
        else:
            images[iImg] = img[idxLow[0]:idxHigh[0]+1, 
                            idxLow[1]:idxHigh[1]+1,
                            idxLow[2]:idxHigh[2]+1]
    
    return images, idxLow, idxHigh


def create_qc_image(fnamesBG, fnameWMH, fnameBmask=None):

    
    # Read images
    if fnameBmask is None:
        fnames = fnamesBG + [fnameWMH] + [fnameWMH]
        onlyZ = True
    else:
        fnames = fnamesBG + [fnameWMH] + [fnameBmask]
        onlyZ = False
    images = []
    for fn in fnames:
        nii = nib.load(fn)
        images.append( nii.get_fdata() )

    # Crop images
    border=None
    images, _, _ = crop_nonzero(images[0:4], images[3], border=border, onlyZ=onlyZ, thr=0, verbose = False)

    # Determine equally spaced slices, whith distance to the image border being half space
    nSlices = 6
    zSize = images[0].shape[2]
    if zSize>16:
        idxSlices = np.ceil(np.round(np.linspace(0,zSize-1, nSlices+1)))
        idxSlices = (idxSlices[0:nSlices] + np.ceil(np.min(np.diff(idxSlices))/2)).astype(np.int32)
    else:
        idxSlices = np.unique(np.round(np.linspace(0,zSize-1, nSlices))).astype(np.int8)
    # print(idxSlices)

    # Extract slices, flip x/y axis, and reshape to 2D image with panels
    axcodes = nib.aff2axcodes(nii.affine)
    for iImg, img in enumerate(images):    
        img = img[0:,0:,idxSlices]
        if axcodes == ('R','A','S'):
            img = np.fliplr(np.rot90(img))
        elif axcodes == ('L','A','S'):
            img = np.rot90(img)
        else:
            print(f'Warning during creation of QC-image: Axis direction codes "{axcodes}" might not be handled correctly!')
        img = img.reshape(img.shape[0],img.shape[1]*nSlices, order='F')
        # images[iImg] = img
        images[iImg] = ndimage.zoom(img, 5, order=0)

    imgMask = np.ma.masked_array(images[2], images[2]==0) 


    # Visualize FLAIR
    percentile=[10,99.75]
    if fnameBmask is not None:
        vlimVol = np.nanpercentile(images[0][images[3]==1], percentile)    
    else:
        vlimVol = np.nanpercentile(images[0], percentile)    
    # create figure with number of pixels corresponding exactly to image shape
    height, width = images[0].shape
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    # Create an axis covering the entire figure space
    ax = fig.add_axes((0, 0, 1, 1))
    # Remove the axes ticks and frames
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    # Plot image and contour
    ax.imshow(images[0], cmap=plt.cm.gray, aspect='equal', interpolation='nearest', vmin=vlimVol[0], vmax=vlimVol[1])
    ax.imshow(imgMask, cmap=plt.cm.bwr, aspect='equal', interpolation='nearest', vmin=0, vmax=1, alpha = 0.2)
    ax.contour(images[2], levels=[0.5], colors='red', linestyles='solid', linewidths=1, alpha=1, antialiased = True)
    if fnameBmask is not None:
        ax.contour(images[3], levels=[0.5], colors='cyan', linestyles='solid', linewidths=2, alpha=1, antialiased = True)
    # Save
    fnames[0] = re.sub(r'\.nii(\.gz)?$', '.png', fnames[0])
    plt.savefig(fnames[0], transparent=False, bbox_inches='tight', pad_inches=0)

    # Visualize T1w
    percentile=[10,99.75]
    if fnameBmask is not None:
        vlimVol = np.nanpercentile(images[1][images[3]==1], percentile)    
    else:
        vlimVol = np.nanpercentile(images[1], percentile)    
    # create figure with number of pixels corresponding exactly to image shape
    # height, width = images[0].shape
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    # Create an axis covering the entire figure space
    ax = fig.add_axes((0, 0, 1, 1))
    # Remove the axes ticks and frames
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    # Plot image and contour
    ax.imshow(images[1], cmap=plt.cm.gray, aspect='equal', interpolation='nearest', vmin=vlimVol[0], vmax=vlimVol[1])
    ax.imshow(imgMask, cmap=plt.cm.bwr, aspect='equal', interpolation='nearest', vmin=0, vmax=1, alpha = 0.2)
    ax.contour(images[2], levels=[0.5], colors='red', linestyles='solid', linewidths=1, alpha=1, antialiased = True)
    if fnameBmask is not None:
        ax.contour(images[3], levels=[0.5], colors='cyan', linestyles='solid', linewidths=2, alpha=1, antialiased = True)
    # Save
    fnames[1] = re.sub(r'\.nii(\.gz)?$', '.png', fnames[1])
    plt.savefig(fnames[1], transparent=True, bbox_inches='tight', pad_inches=0)


    # Create an animated PNG using imageio
    frames = [iio.imread(fn) for fn in fnames[0:2]]
    fnPNG = re.sub(r'\.nii(\.gz)?$', '_QC.png', fnameWMH)
    iio.imwrite(fnPNG, frames, format="PNG", duration=1000)

    return fnPNG
