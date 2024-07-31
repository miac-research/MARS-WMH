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
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /libs \
 && ldd /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit | grep -o '/build[^ ]*' | xargs -I '{}' cp -v '{}' /libs/
 
# Stage 2 
FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3 AS final
RUN adduser --system --no-create-home nonroot

LABEL org.opencontainers.image.source="https://github.com/miac-research/dl-wmh"
LABEL version="1.0.0"
LABEL maintainer="MIAC"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Install Python dependencies
RUN pip3 install --no-cache-dir \
	nibabel==3.2.1 \
	pydicom==2.2.2 \
	pynrrd==0.4.2 \
	scikit-image==0.19.1 \
	setproctitle==1.2.2 \
	visdom==0.1.8.9 \
	pandas==1.5.2 \
 && pip3 install --no-cache-dir --no-deps git+https://github.com/spezold/mvloader.git@4244ba3 \
 && pip3 install --no-cache-dir --no-deps --force-reinstall \
	future==0.18.2 \
	SimpleITK==2.0.1

# Clone MD-GRU
WORKDIR /opt
RUN git clone https://github.com/miac-research/mdgru.git
WORKDIR /opt/mdgru
RUN git reset --hard 791ee4f

# Download MD-GRU model
WORKDIR /model
COPY model/WMH_170000.data-00000-of-00001 .
COPY model/WMH_170000.index .
COPY model/WMH_170000.meta .

# HD-BET
WORKDIR /opt
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
    git=1:2.25.1-1ubuntu3.10 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/MIC-DKFZ/HD-BET \
 && cd HD-BET \
 && git reset --hard d517276 \
 && rm HD_BET/paths.py \
 && echo "folder_with_parameter_files = '/opt/HD-BET_models'" > HD_BET/paths.py \
 && python -m pip install --no-cache-dir -e . \
 && mkdir /opt/HD-BET_models \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/0.model https://zenodo.org/record/2540695/files/0.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/1.model https://zenodo.org/record/2540695/files/1.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/2.model https://zenodo.org/record/2540695/files/2.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/3.model https://zenodo.org/record/2540695/files/3.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/4.model https://zenodo.org/record/2540695/files/4.model?download=1

# Slicer BRAINSfit from Stage 1
WORKDIR /opt/BRAINSFit
COPY --from=builder /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit \
                    /libs ./
ENV LD_LIBRARY_PATH=/opt/BRAINSFit

# Download Python pipeline script
WORKDIR /opt/scripts/
COPY files/pipeline_mdgru.py /opt/scripts/pipeline_mdgru.py

# Entrypoint
USER nonroot
ENTRYPOINT [ "python", "/opt/scripts/pipeline_mdgru.py" ]
