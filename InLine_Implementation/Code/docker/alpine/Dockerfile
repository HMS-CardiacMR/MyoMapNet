FROM python:3.8.2-alpine3.11 AS mrd_converter

RUN     apk --no-cache add hdf5 hdf5-dev --repository http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
     && apk --no-cache add cmake build-base boost-dev boost-static libxml2-dev libxslt-dev git fftw-dev

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
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git

RUN cd /opt/code/siemens_to_ismrmrd && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM python:3.8.2-alpine3.11

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server (Alpine Linux)"
LABEL org.opencontainers.image.url="https://github.com/kspaceKelvin/python-ismrmrd-server"
LABEL org.opencontainers.image.authors="Kelvin Chow (kelvin.chow@siemens-healthineers.com)"

# Copy siemens_to_ismrmrd from last stage and re-add necessary dependencies
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd
COPY --from=mrd_converter /usr/local/lib/libismrmrd*        /usr/local/lib/
RUN     apk --no-cache add hdf5 --repository http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
     && apk --no-cache add libxslt

# Dependencies for Python MRD server
RUN     apk --no-cache add hdf5 hdf5-dev --repository http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
     && apk --no-cache add bash git openssh gcc libc-dev libxslt-dev libxml2-dev \
     && pip install --no-cache-dir h5py

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python.git \
     && cd /opt/code/ismrmrd-python \
     && pip3 install --no-cache-dir .

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && pip3 install --no-cache-dir .

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
RUN apt-get install -y dos2unix \
    && find /opt/code/python-ismrmrd-server -name *.sh | xargs dos2unix \
    && apt-get remove dos2unix -y

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name *.sh -exec chmod +x {} \;

# Cleanup files not required after compiling
RUN  apk del gcc git \
     && rm -r /root/.cache/pip

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
