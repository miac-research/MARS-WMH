# MARS-WMH: Deep learning-based white matter hyperintensity segmentation

The **MIAC Automated Region Segmentation (MARS) for White Matter Hyperintensities (WMH)** is a state-of-the-art, deep learning-based segmentation tool that has undergone systematic validation, both technically and clinically.

This repository includes ready-to-use, pre-built container images of two methods, based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) or [MD-GRU](https://github.com/zubata88/mdgru), along with the code needed to build these images.

The methods are described in detail in the publication currently available as a pre-print:
 
> Gesierich et al., Technical and Clinical Validation of a Novel Deep Learning-Based White Matter Hyperintensity Segmentation Tool. Available as a pre-print at SSRN: [DOI](https://dx.doi.org/10.2139/ssrn.5330929)

Please ensure to cite this publication when using the methods, and please note that the license does not cover any commercial use (defined as use for which any financial return is received). Please also cite the underlying deep learning method (nnU-Net, DOI: [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z) or MD-GRU, DOI: [10.1007/978-3-319-75238-9_3](https://doi.org/10.1007/978-3-319-75238-9_3)).

> [!CAUTION]
> These methods are **NOT medical devices** and **for non-commercial, academic research use only!** 
> Do NOT use these methods for diagnosis, prognosis, monitoring or any other purposes in clinical use.

## Using the pre-built container images

Ready-to-use, pre-built images are available for download from the [Github container registry](https://github.com/miac-research/MARS-WMH/packages). The images have been tested with Apptainer and Docker. 

In general, we recommend the nnU-Net algorithm (please see our publication for a detailed comparison between the two algorithms) and using Apptainer (the standard container tool for scientific computing).

> [!NOTE] 
> Segmentation results from various container versions may differ slightly. Please avoid mixing versions in your project. Additionally, using different compute hardware (CPU versus GPU or varying GPU architectures) may lead to minor variations in the segmentation results.

### Data requirements

The WMH segmentation requires two inputs, a **FLAIR image and a T1-weighted image**, both in NIfTI-1 data format. We recommend [dcm2niix](https://github.com/rordenlab/dcm2niix) for DICOM to NIfTI conversion. Ensure that the entire brain is covered by the field of view. By default, the T1-weighted image will be registered to the FLAIR. However, already registered images can be provided as well (see option `-skipRegistration`).  
- **FLAIR image**: A 3D FLAIR with 1 mm isotropic resolution is recommended.§D images with different resolution will be resliced to 1mm isotropic before prediction, while WMH masks are returned in the original resolution. A 2D FLAIR with up to 3 mm slice thickness can also be used. Images with thicker slices are not recommended.  
- **T1-weighted image**: The recommended resolution is 1 mm isotropic. The image must have been acquired without the use of a contrast agent.

### Hardware requirements

While the inference can be run on CPU (>8 cores recommended), an NVIDIA GPU will greatly accelerate the calculation. The pre-built images support a wide range of NVIDIA GPUs from compute capability 5.0 (Maxwell, 2014) to 12.0 (Blackwell, 2024), please see the table for details:

| NVIDIA CUDA Compute Capability          | nnU-Net image | MD-GRU image |
| --------------------------------------- | ------------- | ------------ |
| 12.0 (Blackwell, 2024)                  | supported     | limited*     |
| 7.5 (Turing, 2018) – 9.0 (Hopper, 2022) | supported     | supported    |
| 5.0 (Maxwell, 2014) – 7.0 (Volta, 2017) | only [v1.0.0](https://github.com/miac-research/MARS-WMH/pkgs/container/wmh-nnunet/391770322?tag=1.0.0) | supported |

*While not officially supported, our testing indicates that the MD-GRU image can run on Blackwell. However, the brainmask will be generated using the CPU, which is much slower and may produce slightly different results compared to an all-GPU pipeline.

### nnU-Net algorithm using Apptainer

```shell
# 1. Pull the container image and save as .sif file 
apptainer build mars-wmh-nnunet.sif docker://ghcr.io/miac-research/wmh-nnunet:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--nv")
apptainer run -B $(pwd) --nv mars-wmh-nnunet.sif --flair FLAIR.nii.gz --t1 T1w.nii.gz

# For advanced usage, see available command line options:
apptainer run mars-wmh-nnunet.sif -h
```

### nnU-Net algorithm using Docker

```shell
# 1. Pull the container image into your local registry
docker pull ghcr.io/miac-research/wmh-nnunet:latest
docker tag ghcr.io/miac-research/wmh-nnunet:latest mars-wmh-nnunet:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--gpus all")
docker run --rm --gpus all -v $(pwd):/data mars-wmh-nnunet:latest --flair /data/FLAIR.nii.gz --t1 /data/T1w.nii.gz

# For advanced usage, see available command line options:
docker run --rm mars-wmh-nnunet:latest -h
```

### MD-GRU algorithm using Apptainer

```shell
# 1. Pull the container image and save as .sif file 
apptainer build mars-wmh-mdgru.sif docker://ghcr.io/miac-research/wmh-mdgru:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--nv")
apptainer run -B $(pwd) --nv mars-wmh-mdgru.sif --flair FLAIR.nii.gz --t1 T1w.nii.gz

# For advanced usage, see available command line options:
apptainer run mars-wmh-mdgru.sif -h
```

### MD-GRU algorithm using Docker

```shell
# 1. Pull the container image into your local registry
docker pull ghcr.io/miac-research/wmh-mdgru:latest
docker tag ghcr.io/miac-research/wmh-mdgru:latest mars-wmh-mdgru:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--gpus all")
docker run --rm --gpus all -v $(pwd):/data mars-wmh-mdgru:latest --flair /data/FLAIR.nii.gz --t1 /data/T1w.nii.gz

# For advanced usage, see available command line options:
docker run --rm mars-wmh-mdgru:latest -h
```

## Building the container images yourself

If you do not want to use the pre-built images, you can build them yourself locally using the provided Dockerfiles in the `mdgru` and `nnunet` folders.

1. Download the mdgru or nnunet Dockerfile and place it into a local folder.
2. In this folder, run `docker build -t mars-wmh-{mdgru/nnunet} .`

> [!NOTE]
> During building, multiple external sources need to be used, e.g., base images are downloaded from the NVIDIA NGC registry, scripts are downloades from this Github repository, and larger model files from Zenodo. Make sure you can access all required external sources in your build environment.

## Licenses of redistributed software

Please note the license terms of software components that we redistribute within our container images:

- [HD-BET](https://github.com/MIC-DKFZ/HD-BET?tab=Apache-2.0-1-ov-file)
- [3D Slicer](https://github.com/Slicer/Slicer/tree/main?tab=License-1-ov-file)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet?tab=Apache-2.0-1-ov-file)
- [MD-GRU](https://github.com/zubata88/mdgru?tab=LGPL-2.1-1-ov-file)

## Funding

Development and maintenance of this software is funded by the [Medical Image Analysis Center (MIAC AG)](https://miac.swiss).

[![MIAC Logo](http://miac.swiss/gallery/normal/116/miaclogo@2x.png)](https://miac.swiss)
