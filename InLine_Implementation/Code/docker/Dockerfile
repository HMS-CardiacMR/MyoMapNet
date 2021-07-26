# ----- First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM python:3.6 AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN  apt-get update && apt-get install -y git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev
RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM python:3.6

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"
LABEL org.opencontainers.image.url="https://github.com/kspaceKelvin/python-ismrmrd-server"
LABEL org.opencontainers.image.authors="Kelvin Chow (kelvin.chow@siemens-healthineers.com)"

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# xslt and hdf5 are dependencies for siemens_to_ismrmrd
RUN apt-get update && apt-get install --no-install-recommends -y libxslt1.1 libhdf5-103 git
RUN mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && pip3 install --no-cache-dir .

# Install dependencies needed
RUN pip3 install --no-cache-dir cached-property==1.5.2
RUN pip3 install --no-cache-dir cycler==0.10.0
RUN pip3 install --no-cache-dir dataclasses==0.8
RUN pip3 install --no-cache-dir h5py==3.1.0
RUN pip3 install --no-cache-dir kiwisolver==1.3.1
RUN pip3 install --no-cache-dir matplotlib==3.3.4
RUN pip3 install --no-cache-dir numpy==1.19.5
RUN pip3 install --no-cache-dir Pillow==8.2.0
RUN pip3 install --no-cache-dir protobuf==3.15.8
RUN pip3 install --no-cache-dir pycparser==2.20
RUN pip3 install --no-cache-dir pydensecrf==1.0rc3
RUN pip3 install --no-cache-dir pyparsing==2.4.7
RUN pip3 install --no-cache-dir python-dateutil==2.8.1
RUN pip3 install --no-cache-dir PyYAML==5.4.1
RUN pip3 install --no-cache-dir scipy==1.5.4
RUN pip3 install --no-cache-dir six==1.15.0
RUN pip3 install --no-cache-dir tensorboardX==2.2
RUN pip3 install --no-cache-dir torch==1.4.0
RUN pip3 install --no-cache-dir torchvision==0.5.0
RUN pip3 install --no-cache-dir xsdata==21.4
RUN pip3 install --no-cache-dir ismrmrd==1.7.2

# # If building from the GitHub repo, uncomment the below section, open a command
# # prompt in the folder containing this Dockerfile and run the command:
# #    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN  cd /opt/code \
#      && git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# If doing local development, use this section to copy local code into Docker
# image. From the folder containing the python-ismrmrd-server repo, uncomment
# the COPY line below and run the command:
#    docker build --no-cache -t fire-python-custom -f python-ismrmrd-server/docker/Dockerfile ./
COPY python-ismrmrd-server  /opt/code/python-ismrmrd-server

# Ensure startup scripts have Unix (LF) line endings, which may not be true
# if the git repo is cloned in Windows
RUN apt-get update && apt-get install -y dos2unix \
    && find /opt/code/python-ismrmrd-server -name *.sh | xargs dos2unix \
    && apt-get remove dos2unix -y

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name *.sh -exec chmod +x {} \;

# Cleanup files not required after installation
RUN  apt-get remove git -y \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* \
     && rm -rf /root/.cache/pip

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
