FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Slicer BRAINSfit Registration Module    
WORKDIR /opt
RUN wget -nv -O Slicer-4.10.2-linux-amd64.tar.gz https://slicer-packages.kitware.com/api/v1/file/60add73aae4540bf6a89c03b/download
RUN tar -zxf Slicer-4.10.2-linux-amd64.tar.gz -C /opt \
 && rm Slicer-4.10.2-linux-amd64.tar.gz \
 && rm    /opt/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/libPythonQt.so \
 && rm    /opt/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/libQt5WebEngineCore.so.5.11.2 \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/python2.7 \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/lib/Python \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/lib/cmake \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/lib/Teem-1.12.0 \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/lib/QtPlugins \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/Slicer \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/bin \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/include \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/libexec \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/resources \
 && rm -r /opt/Slicer-4.10.2-linux-amd64/share 
ENV LD_LIBRARY_PATH=/opt/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10:/opt/Slicer-4.10.2-linux-amd64/lib/Slicer-4.10/cli-modules

# HD-BET
RUN apt-get update -qq && apt-get install -y -q --no-install-recommends git
WORKDIR /opt
RUN git clone https://github.com/MIC-DKFZ/HD-BET \
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
 
# Download Python pipeline script
WORKDIR /opt/scripts/
# RUN wget -nv -O pipeline_mdgru.py https://raw.githubusercontent.com/miac-research/dl-brainstem/main/mdgru/pipeline_mdgru.py
COPY files/pipeline_mdgru.py /opt/scripts/pipeline_mdgru.py

# Entrypoint
ENTRYPOINT [ "python", "/opt/scripts/pipeline_mdgru.py" ]
