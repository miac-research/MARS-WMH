# Container images for deep learning-based WMH segmentation

This repository contains the code required to build the container images of two **deep learning-based brain white matter hyperintensity (WMH) segmentation methods**, based on nnU-net or MD-GRU.

The methods are described in detail in the following publication: 
 
> Submitted. To be added.

Please make sure to cite this publication when using the methods, and please note that the [license](https://github.com/miac-research/dl-wmh/blob/main/LICENSE) does not cover any commercial use (defined as use for which any financial return is received).

> [!CAUTION]
> These methods are **NOT a medical device** and **for non-commercial, academic research use only!**  
> Do NOT use these methods for diagnosis, prognosis, monitoring or any other purposes in clinical use.

## Using the pre-built container images

Ready-to-use, pre-built images for nnU-Net and MD-GRU WMH segmentaion are available for download from the [Github container registry](https://github.com/miac-research/dl-wmh/packages). The images have been tested with Apptainer and Docker. 

In general, we recommend the nnU-Net algorithm (please see our publication for a detailed comparison between the two algorithms) and using Apptainer (the standard container tool for scientific computing).

### Hardware requirements

While the inference can be run on CPU (>8 cores recommended), an NVIDIA GPU will greatly accelerate the calculation. The pre-built images use CUDA 12 and can thus support a wide range of NVIDIA GPUs from compute capability 5.0 (Maxwell generation, 2014) to 9.0 (current generation). A minimum of 8 GB GPU memory is required.

### nnU-Net algorithm using Apptainer

```shell
# 1. Pull the container image and save as .sif file   
apptainer build wmh-nnunet.sif docker://ghcr.io/miac-research/wmh-nnunet:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--nv")
apptainer run -B $(pwd) --nv wmh-nnunet.sif --flair FLAIR.nii.gz --t1 T1w.nii.gz

# For advanced usage, see available command line options:
apptainer run wmh-nnunet.sif -h
```

### nnU-Net algorithm using Docker

```shell
# 1. Pull the container image into your local registry
docker pull ghcr.io/miac-research/wmh-nnunet:latest
docker tag ghcr.io/miac-research/wmh-nnunet:latest wmh-nnunet:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--gpus all")
docker run --rm --gpus all -v $(pwd):/data  wmh-nnunet:latest --flair /data/FLAIR.nii.gz --t1 /data/T1w.nii.gz

# For advanced usage, see available command line options:
docker run --rm wmh-nnunet:latest -h
```

### MD-GRU algorithm using Apptainer

```shell
# 1. Pull the container image and save as .sif file   
apptainer build wmh-mdgru.sif docker://ghcr.io/miac-research/wmh-mdgru:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--nv")
apptainer run -B $(pwd) --nv wmh-mdgru.sif --flair FLAIR.nii.gz --t1 T1w.nii.gz

# For advanced usage, see available command line options:
apptainer run wmh-mdgru.sif -h
```

### MD-GRU algorithm using Docker

```shell
# 1. Pull the container image into your local registry
docker pull ghcr.io/miac-research/wmh-mdgru:latest
docker tag ghcr.io/miac-research/wmh-mdgru:latest wmh-mdgru:latest

# 2. Run inference on a pair of FLAIR and T1w images in the current working directory using GPU (flag "--gpus all")
docker run --rm --gpus all -v $(pwd):/data  wmh-mdgru:latest --flair /data/FLAIR.nii.gz --t1 /data/T1w.nii.gz

# For advanced usage, see available command line options:
docker run --rm wmh-mdgru:latest -h
```

## Building the container images yourself

If you do not want to use the pre-built images, you can build them yourself locally using the provided Dockerfiles in the `mdgru` and `nnunet` folders.

1. Download the mdgru or nnunet Dockerfile and place it into a local folder.
2. In this folder, run `docker build -t wmh-{mdgru/nnunet} .`

> [!NOTE]
> During building, multiple external sources need to be used, e.g., base images are downloaded from the NVIDIA NGC registry, scripts are download from this Github repository, and larger model files are downloaded from Zenodo. Make sure you can access all required external sources in your build environment.
