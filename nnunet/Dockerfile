# Stage 1
FROM ubuntu:noble-20250619 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        python-is-python3=3.11.4* \
        python3=3.12.3* \
        python3.12-venv=3.12.3* \
        python3-pip=24.0* \
        wget=1.21.4* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # Slicer BRAINSfit
WORKDIR /build
RUN wget --progress=dot:giga --no-check-certificate -O Slicer-4.10.2-linux-amd64.tar.gz https://slicer-packages.kitware.com/api/v1/file/60add73aae4540bf6a89c03b/download \
 && tar -zxf Slicer-4.10.2-linux-amd64.tar.gz -C /build
ENV LD_LIBRARY_PATH=/build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10:/build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /libs \
 && ldd /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit | grep -o '/build[^ ]*' | xargs -I '{}' cp -v '{}' /libs/

# Pytorch, nnU-Net
RUN python -m venv /venv \
 && /venv/bin/pip install --upgrade pip \
 && /venv/bin/pip install --no-cache-dir --prefer-binary \
        torch --index-url https://download.pytorch.org/whl/cu128 \
 && /venv/bin/pip install --no-cache-dir --prefer-binary \
        setuptools==69.0.3 \
        wheel==0.42.0 \
        nnunet==1.7.1

# Modify nnunet model_restore.py to revert new default setting in torch==2.6 and newer
RUN sed -i 's/map_location=torch.device('\''cpu'\'')/map_location=torch.device('\''cpu'\''), weights_only=False/g' /venv/lib/python3.12/site-packages/nnunet/training/model_restore.py

# HD-BET
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        git \
 && git clone https://github.com/MIC-DKFZ/HD-BET \
 && cd HD-BET \
 && git reset --hard 0dcab33233e991b9f7a7cb33052b164eb50062bd \
 && rm HD_BET/paths.py \
 && echo "folder_with_parameter_files = '/opt/HD-BET_models'" > HD_BET/paths.py \
 && /venv/bin/pip install --no-cache-dir . \
 && sed -i '1i #!/venv/bin/python' /venv/bin/hd-bet \
 && sed -i '/^import imp$/d' /venv/lib/python3.12/site-packages/HD_BET/run.py

# venv finalization and clean-up
RUN /venv/bin/pip install --no-cache-dir --prefer-binary --force-reinstall \
        numpy==1.26.4 \
 && find /venv -type d -name '__pycache__' -exec rm -rf {} + \
 && find /venv -type f -name '*.pyc' -delete


# Stage 2 
FROM ubuntu:noble-20250619 AS final

LABEL org.opencontainers.image.authors="https://miac.swiss" \
    org.opencontainers.image.source="https://github.com/miac-research/MARS-WMH" \
    org.opencontainers.image.url="https://github.com/miac-research/MARS-WMH" \
    org.opencontainers.image.description="Ready-to-use container image for MARS-WMH using nnU-Net" \
    org.opencontainers.image.version="1.0.2" \
    version="1.0.2"

ENV nnUNet_raw_data_base=/nnunet/raw \
    nnUNet_preprocessed=/nnunet/preprocessed \
    RESULTS_FOLDER=/nnunet/results \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/venv \
    PATH="/venv/bin:$PATH"

# OS dependencies, add non-root user, fix locale
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends \
        adduser=3.137* \
        locales=2.39* \
        python-is-python3=3.11.4* \
        python3=3.12.3* \
        python3-pip=24.0* \
        wget=1.21.4* \
 && adduser --system --no-create-home nonroot \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && dpkg-reconfigure --frontend=noninteractive locales \
 && update-locale LANG="en_US.UTF-8" \
 && apt-get purge -y --auto-remove \
        adduser \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create folder structure required by nnU-Net
RUN mkdir -p /nnunet/raw/nnUNet_cropped_data \
        /nnunet/raw/nnUNet_raw_data \
        /nnunet/preprocessed \
        /nnunet/results

# nnU-Net model
WORKDIR /nnunet/results/nnUNet/3d_fullres/Task700_WMH/nnUNetTrainerV2__nnUNetPlansv2.1_8GB_iso1mm
RUN wget -nv -O nnUNet_wmh.tar.gz https://zenodo.org/records/15173303/files/nnUNet_wmh.tar.gz?download=1 \
 && tar xvf nnUNet_wmh.tar.gz -C ./ \
 && rm nnUNet_wmh.tar.gz

# HD-BET models
WORKDIR /opt/HD-BET_models
RUN wget -nv --no-check-certificate -O /opt/HD-BET_models/0.model https://zenodo.org/record/2540695/files/0.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/1.model https://zenodo.org/record/2540695/files/1.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/2.model https://zenodo.org/record/2540695/files/2.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/3.model https://zenodo.org/record/2540695/files/3.model?download=1 \
 && wget -nv --no-check-certificate -O /opt/HD-BET_models/4.model https://zenodo.org/record/2540695/files/4.model?download=1

# Slicer BRAINSfit from builder stage
WORKDIR /opt/BRAINSFit
COPY --from=builder /build/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules/BRAINSFit \
                    /libs ./
ENV LD_LIBRARY_PATH=/opt/BRAINSFit

# Copy venv from builder stage
COPY --from=builder /venv /venv 

# MARS scripts
WORKDIR /opt/scripts/
COPY files/ .

USER nonroot
ENTRYPOINT [ "python", "/opt/scripts/pipeline_nnunet.py" ]
