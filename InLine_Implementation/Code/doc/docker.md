### Getting Started with Docker
Docker is a virtualization platform that allows software to run in isolated environments called containers.  It provides a convenient mechanism to package up a reconstruction program and all its libraries in a manner that can be easily deployed to other computers without manually installing dependencies or other configuration steps.  

Install Docker from https://www.docker.com/products/docker-desktop with the standard settings.  Note that Docker is not compatible with Hyper-V, which is required for VirtualBox.  Both Docker and VirtualBox can be installed at the same time, but only one can be usable depending on whether Hyper-V is enabled or not.  Hyper-V can be turned on/off following the instructions at https://docs.microsoft.com/en-us/troubleshoot/windows-client/application-management/virtualization-apps-not-work-with-hyper-v.  A reboot is required for the change to take effect.

Download the Docker image of this server by opening a command prompt and running:
```
docker pull kspacekelvin/fire-python
```

#### Start the Reconstruction Server in a Docker Container
If using Windows, create the folder ``C:\tmp``.  In a command prompt, run the command:
```
In Windows:
    docker run -p=9002:9002 --rm -it -v C:\tmp:/tmp kspacekelvin/fire-python

In MacOS/Linux:
    docker run -p=9002:9002 --rm -it -v /tmp:/tmp kspacekelvin/fire-python
```

The command line options used are:
```
-p=9002:9002    Allows access to port 9002 inside the container from port 9002 on the host.  Change
                the first number to change the host port.
-it             Enables “interactive” mode with a pseudo-tty.  This is necessary for “ctrl-c” to
                stop the program.
--rm            Remove the container after it is stopped
-v /tmp:/tmp    Maps the /tmp folder on the host to /tmp inside the container.  Change the first
                path to change the host folder.  Log and debug files are stored in this folder.
```

The server can be stopped by pressing ``ctrl-c``.

#### Start the Reconstruction Client
In a separate command prompt, start another container of the Docker image:
```
docker run --rm -it -v /tmp:/tmp kspacekelvin/fire-python /bin/bash
```

In this command, the ``/bin/bash`` argument is used to start the container with a bash shell prompt instead of starting the Python MRD server.  Within this bash shell, generate a sample raw data set:
```
python3 /opt/code/ismrmrd-python-tools/generate_cartesian_shepp_logan_dataset.py -o /tmp/phantom_raw.h5
```

Run the client and send data to the server in the other Docker container:
```
python3 /opt/code/python-ismrmrd-server/client.py -a 172.17.0.1 -p 9002 -G "dataset" -o /tmp/phantom_img.h5 /tmp/phantom_raw.h5
```

The command line options used are:
```
-a 172.17.0.1  Send data to address 172.17.0.1, which is the default IP address of the Docker host.
-p 9002        Send data to port 9002.  This is sent to the host, which is then redirected to the
               server container due to the “-p” port mapping when starting the server. 
-o             Specifies the output file name
-G             Specifies the group name in the output file

The last argument specifies the input file name.
```

MRD image data are also stored in HDF files arranged by groups as named by the ``-G`` argument.  If this argument is omitted, a group name with the current date/time is used.  Images are further grouped by series index, with a sub-group named ``image_x``, where x is ``image_series_index`` in the [ImageHeader](https://ismrmrd.github.io/apidocs/1.4.2/struct_i_s_m_r_m_r_d_1_1_i_s_m_r_m_r_d___image_header.html).  For example:
```
/dataset/image_0/data         Image data
/dataset/image_0/header       MRD ImageHeader structure
/dataset/image_0/attributes   MRD MetaAttributes text
```

As Docker provides only a command-line virtualization interface, it not possible to directly view the reconstructed images from within the Docker container.  However, the output file stored in ``C:\tmp`` or ``\tmp`` on the host and can be read using [ismrmrdviewer](https://github.com/ismrmrd/ismrmrdviewer) or [HDFView](https://www.hdfgroup.org/downloads/hdfview/).


### Building a Docker Image
A [Dockerfile](../docker/Dockerfile) is provided using based on [python:3.9-slim](https://github.com/docker-library/python/blob/18ef1b6500a622da8b138d1b413c963c279e7ea4/3.9/buster/slim/Dockerfile), a light Python image optimized for reduced total size.  A two-stage build is also used to include the [ismrmrd](https://github.com/ismrmrd/ismrmrd) and [siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd) packages without needing their build dependencies in the final image.

For some image analysis code, additional packages or libraries may be required.  To create a Docker image with these additional packages, modify the Dockerfile by adding ``RUN`` commands corresponding to how the packages would be installed via command line.  Temporary files created during each ``RUN`` command are kept in the final image, so group installations of multiple packages from the same manager (e.g. apt or pip) whenever possible.

When determining the required packages, an interactive approach is often useful.  Build the existing Dockerfile by opening a command prompt in the folder containing the python-ismrmrd-server repo:
```
docker build --no-cache -t fire-python -f python-ismrmrd-server/docker/Dockerfile ./
```

The above command uses the following options:
```
--no-cache                                  Run each step of the Docker build process without caching from previous builds
-t fire-python                              Tag (name) of the Docker image.
-f python-ismrmrd-server/docker/Dockerfile  Path to the Dockerfile
./                                          Build context (folder from which COPY commands use as root)
```

Start an instance of this image:
```
docker run -it --rm fire-python
```

The following options are used:
```
-it     Use an interactive terminal with command line input
--rm    Delete the container after exiting
```

Within this container, attempt to start the Python server by running ``python3 /opt/code/python-ismrmrd-server/main.py``.  If any errors are encountered, install the libraries using the appropriate apt, pip, or other commands.  Copy these commands into the Dockerfile as each dependency is resolved.  It may be necessary to send some data to the server to ensure any run-time dependencies are also validated.  To do so, a ``-p 9002:9002`` command is required during the ``docker run`` step in order to share the port.  More details about running a client/server pair in Docker are available here. Once all dependencies are resolved, exit the Docker container by running ``quit`` at the command prompt, rebuild the Docker image, and ensure that you can start the server with ``docker run`` without any additional steps.

### Creation of a chroot image
A chroot image contains the complete contents of a Linux operating system and serves as the root folder for the reconstruction program.  The chroot image can contain libraries and other files that can be used by the reconstruction program, isolated from the Linux operating system on the MARS computer.  Operating systems tested for FIRE chroot compatibility include Ubuntu, Debian, and Alpine.  Chroot images can be generated using manual tools such as debootstrap or be created from existing containers such as Docker.

A Docker image of the MRD reconstruction server in this repository can be found on Docker Hub.  It can be downloaded by running:
```
docker pull kspacekelvin/fire-python
```

A Docker image using the MRD reconstruction server in this repository can be created using the provided [Dockerfile](../docker/Dockerfile).  To build it, run the following command from the folder containing the python-ismrmrd-server repo folder:
```
docker build --no-cache -t kspacekelvin/fire-python -f python-ismrmrd-server/docker/Dockerfile ./
```

The following steps can be used to create a chroot image from a Docker image.  They are performed within a Linux Docker image, but can also be run on a Linux system natively.
1. Create a Docker container instance from an image.  If a different tag was used above, change the last argument accordingly.
    ```
    docker create --name tmpimage kspacekelvin/fire-python
    ```

2. Export the file system contents to a tar archive.  Create the tmp folder if necessary.
    ```
    In Windows:
        docker export -o C:\tmp\fire-python-contents.tar tmpimage

    In MacOS/Linux:
        docker export -o /tmp/fire-python-contents.tar tmpimage
    ```

3. Remove the Docker container instance.
    ```
    docker rm tmpimage
    ```

4. Start a Ubuntu Linux Docker container, sharing the tmp folder.
    ```
    In Windows:
        docker run -it --rm --privileged=true -v C:\tmp:/tmp ubuntu

    In MacOS/Linux:
        docker run -it --rm --privileged=true -v /tmp:/tmp ubuntu
    ```

    The following options are used:
    ```
    -it                Use an interactive terminal with command line input
    --rm               Delete the container after exiting
    --privileged=true  Use extended privileges to allow mount commands
    -v /tmp:/tmp       Share volume (folder) from host to container
    ```

5. Create a blank chroot file with an ext3 file system 425 MB in size.  The total file size is the product of the number of blocks (count) and the block size (bs).  However, the available space is ~30 MB less than the file size due to file system overhead.  The available space must be greater than the size of the tar archive above, with sufficient additional space (~10%) for temporary files that may be created during image reconstruction.
    ```
    dd if=/dev/zero of=/tmp/fire-python-chroot.img bs=1M count=425
    mke2fs -F -t ext3 /tmp/fire-python-chroot.img
    ```

6. Mount the chroot file.  If not using Docker, add “sudo” before the mount command.
    ```
    mkdir /mnt/chroot
    mount -o loop /tmp/fire-python-chroot.img /mnt/chroot
    ```

7. Extract the image contents into the mounted chroot image.
    ```
    tar -xvf /tmp/fire-python-contents.tar --directory=/mnt/chroot
    ```

8. Verify the amount of free space available on the chroot image (52 MB in the below):
    ```
    root@0cdce2f7e3cf:/# df -h
    Filesystem      Size  Used Avail Use% Mounted on
    /dev/loop0      396M  324M   52M  87% /mnt/chroot
    ```

9. Unmount the chroot image.
    ```
    umount /mnt/chroot
    ```

10. (Optional) The chroot is highly compressible using the zip file format.
    ```
    zip -j /tmp/fire-python-chroot.zip /tmp/fire-python-chroot.img
    ```

11.  Exit the Docker container instance if started in step 4.
    ```
    exit
    ```
