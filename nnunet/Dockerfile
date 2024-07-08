# Stage 1: Slicer BRAINSFit with dependencies
FROM ubuntu:jammy-20240627.1 AS builder

RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
    wget=1.21.2-2ubuntu1.1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /build 
RUN wget --progress=dot:giga --no-check-certificate -O Slicer-4.10.2-linux-amd64.tar.gz https://slicer-packages.kitware.com/api/v1/file/60add73aae4540bf6a89c03b/download \
 && tar -zxf Slicer-4.10.2-linux-amd64.tar.gz -C /build
ENV LD_LIBRARY_PATH=/build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10:/build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules
RUN mkdir -p /libs \
 && ldd /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit | grep -o '/build[^ ]*' | xargs -I '{}' cp -v '{}' /libs/
 
# Stage 2 
FROM nvcr.io/nvidia/cuda:12.1.1-base-ubuntu22.04 AS final

LABEL org.opencontainers.image.source https://github.com/miac-research/dl-wmh
LABEL version="1.0.0"
LABEL maintainer="MIAC"

# Create folder structure required by nnU-Net
RUN mkdir -p /nnunet/raw \
 && mkdir -p /nnunet/raw/nnUNet_cropped_data \
 && mkdir -p /nnunet/raw/nnUNet_raw_data \
 && mkdir -p /nnunet/preprocessed \
 && mkdir -p /nnunet/results

ENV nnUNet_raw_data_base=/nnunet/raw \
    nnUNet_preprocessed=/nnunet/preprocessed \
    RESULTS_FOLDER=/nnunet/results \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install OS dependencies
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
	python3=3.10.6-1~22.04 \
	python3-pip=22.0.2+dfsg-1ubuntu0.4 \
	python-is-python3=3.9.2-2 \
    wget=1.21.2-2ubuntu1.1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==24.1.1 \
 && pip3 install --no-cache-dir \
	setuptools==66.1.1 \
 	wheel==0.42.0 \
	torch==2.1.1 \
	nnunet==1.7.1 \
 && pip3 install --no-cache-dir --no-deps --force-reinstall \
    numpy==1.23.5
    
# nnU-Net model
WORKDIR /nnunet/results/nnUNet/3d_fullres/Task700_WMH/nnUNetTrainerV2__nnUNetPlansv2.1_8GB_iso1mm
COPY files/nnUNet_wmh .

# HD-BET
WORKDIR /opt
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends git \
 && git clone https://github.com/MIC-DKFZ/HD-BET \
 && cd HD-BET \
 && git reset --hard d517276 \
 && rm HD_BET/paths.py \
 && echo "folder_with_parameter_files = '/opt/HD-BET_models'" > HD_BET/paths.py \
 && python -m pip install --no-cache-dir -e .

WORKDIR /opt/HD-BET_models
RUN wget -nv -O 0.model https://zenodo.org/record/2540695/files/0.model?download=1 \
 && wget -nv -O 1.model https://zenodo.org/record/2540695/files/1.model?download=1 \
 && wget -nv -O 2.model https://zenodo.org/record/2540695/files/2.model?download=1 \
 && wget -nv -O 3.model https://zenodo.org/record/2540695/files/3.model?download=1 \
 && wget -nv -O 4.model https://zenodo.org/record/2540695/files/4.model?download=1

# Slicer BRAINSfit from Stage 1
WORKDIR /opt/BRAINSFit
COPY --from=builder /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit .
COPY --from=builder /libs .
ENV LD_LIBRARY_PATH=/opt/BRAINSFit

# Download Python pipeline script
WORKDIR /opt/scripts/
COPY files/pipeline_nnunet.py .

# Entrypoint
ENTRYPOINT [ "python", "/opt/scripts/pipeline_nnunet.py" ]
