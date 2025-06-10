# Stage 1: Slicer BRAINSFit with dependencies
FROM ubuntu:jammy-20240627.1 AS builder

RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        wget=1.21.2-2ubuntu1.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 
WORKDIR /build 
RUN wget --progress=dot:giga --no-check-certificate -O Slicer-4.10.2-linux-amd64.tar.gz https://slicer-packages.kitware.com/api/v1/file/60add73aae4540bf6a89c03b/download && \
    tar -zxf Slicer-4.10.2-linux-amd64.tar.gz -C /build
ENV LD_LIBRARY_PATH=/build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10:/build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /libs && \
    ldd /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit | grep -o '/build[^ ]*' | xargs -I '{}' cp -v '{}' /libs/
 
# Stage 2 
FROM nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS final

LABEL org.opencontainers.image.authors="https://miac.swiss"
LABEL org.opencontainers.image.source="https://github.com/miac-research/MARS-WMH"
LABEL org.opencontainers.image.url="https://github.com/miac-research/MARS-WMH"
LABEL org.opencontainers.image.description="Ready-to-use container image for MARS-WMH using nnU-Net"
LABEL org.opencontainers.image.version="1.0.1"
LABEL version="1.0.1"

# Set environment variables
ENV nnUNet_raw_data_base=/nnunet/raw \
    nnUNet_preprocessed=/nnunet/preprocessed \
    RESULTS_FOLDER=/nnunet/results \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Install OS dependencies and fix locale
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        locales=2.35* \
        python3=3.10.6* \
        python3-pip=22.0.2* \
        python-is-python3 \
        wget=1.21.2* && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG="en_US.UTF-8" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create folder structure required by nnU-Net
RUN mkdir -p /nnunet/raw/nnUNet_cropped_data \
        /nnunet/raw/nnUNet_raw_data \
        /nnunet/preprocessed \
        /nnunet/results

# Install Python dependencies including torch with CUDA 12.8
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip==23.3.2 && \
    python3 -m pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --no-cache-dir \
        setuptools==69.0.3 \
        wheel==0.42.0 \
        nnunet==1.7.1 && \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        numpy==1.23.5

# Modify nnunet model_restore.py to revert new default setting in torch==2.6 and newer
RUN sed -i 's/map_location=torch.device('\''cpu'\'')/map_location=torch.device('\''cpu'\''), weights_only=False/g' /usr/local/lib/python3.10/dist-packages/nnunet/training/model_restore.py

# nnU-Net model
WORKDIR /nnunet/results/nnUNet/3d_fullres/Task700_WMH/nnUNetTrainerV2__nnUNetPlansv2.1_8GB_iso1mm
RUN wget -nv -O nnUNet_wmh.tar.gz https://zenodo.org/records/15173303/files/nnUNet_wmh.tar.gz?download=1 && \
    tar xvf nnUNet_wmh.tar.gz -C ./ && \
    rm nnUNet_wmh.tar.gz

# HD-BET
WORKDIR /opt
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        git=1:2.34.1-1ubuntu1.12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/MIC-DKFZ/HD-BET && \
    cd HD-BET && \
    git reset --hard d517276 && \
    rm HD_BET/paths.py && \
    echo "folder_with_parameter_files = '/opt/HD-BET_models'" > HD_BET/paths.py && \
    python -m pip install --no-cache-dir -e . && \
    mkdir /opt/HD-BET_models && \
    wget -nv --no-check-certificate -O /opt/HD-BET_models/0.model https://zenodo.org/record/2540695/files/0.model?download=1 && \
    wget -nv --no-check-certificate -O /opt/HD-BET_models/1.model https://zenodo.org/record/2540695/files/1.model?download=1 && \
    wget -nv --no-check-certificate -O /opt/HD-BET_models/2.model https://zenodo.org/record/2540695/files/2.model?download=1 && \
    wget -nv --no-check-certificate -O /opt/HD-BET_models/3.model https://zenodo.org/record/2540695/files/3.model?download=1 && \
    wget -nv --no-check-certificate -O /opt/HD-BET_models/4.model https://zenodo.org/record/2540695/files/4.model?download=1 && \
    pip3 install --no-cache-dir --no-deps --force-reinstall \
    numpy==1.24.4

# Slicer BRAINSfit from Stage 1
WORKDIR /opt/BRAINSFit
COPY --from=builder /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit \
                    /libs ./
ENV LD_LIBRARY_PATH=/opt/BRAINSFit

# Download Python pipeline script
WORKDIR /opt/scripts/
RUN wget -nv -O pipeline_nnunet.py https://raw.githubusercontent.com/miac-research/MARS-WMH/refs/heads/main/nnunet/files/pipeline_nnunet.py && \
    wget -nv -O create_qc_image.py https://raw.githubusercontent.com/miac-research/MARS-WMH/refs/heads/main/nnunet/files/create_qc_image.py && \
    wget -nv -O create_html_with_png.py https://raw.githubusercontent.com/miac-research/MARS-WMH/refs/heads/main/nnunet/files/create_html_with_png.py

# Add nonroot user and switch to it
RUN adduser --system --no-create-home nonroot
USER nonroot

ENTRYPOINT [ "python", "/opt/scripts/pipeline_nnunet.py" ]
