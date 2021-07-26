### Code Design
This code is designed to provide a reference implementation of an MRD client/server pair.  It is modular and can be easily extended to include additional reconstruction/analysis programs.

- [main.py](main.py):  This is the main program, parsing input arguments and starting a "Server" class instance.

- [server.py](server.py):  The “Server” class determines which reconstruction algorithm (e.g. "simplefft" or "invertcontrast") is used for the incoming data, based on the requested config information.

- [connection.py](connection.py): The “Connection” class handles network communications to/from the client, parsing streaming messages of different types, as detailed in the section on MR Data Message Format.  The connection class also saves incoming data to MRD files if this option is selected.

- [constants.py](constants.py): This file contains constants that define the message types of the MRD streaming data format.

- [mrdhelper.py](mrdhelper.py): This class contains helper functions for commonly used MRD tasks such as copying header information from raw data to image data and working with image metadata.

- [simplefft.py](simplefft.py): This file contains code for performing a rudimentary image reconstruction from raw data, consisting of a Fourier transform, sum-of-squares coil combination, signal intensity normalization, and removal of phase oversampling.

- [invertcontrast.py](invertcontrast.py): This program accepts both incoming raw data as well as image data.  The image contrast is inverted and images are sent back to the client.

- [rgb.py](rgb.py): This program accepts incoming image data, applies a jet colormap, and sends RGB images back to the client.

- [analyzeflow.py](analyzeflow.py): This program accepts velocity phase contrast image data and performs basic masking.

- [client.py](client.py): This script can be used to function as the client for an MRD streaming session, sending data from a file to a server and saving the received images to a different file.  Additional description of its usage is provided below.

- [generate_cartesian_shepp_logan_dataset.py](generate_cartesian_shepp_logan_dataset.py): Creates an MRD raw data file of a Shepp-Logan phantom with Cartesian sampling.  Borrowed from the [ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools) repository.

- [mrd2gif.py](mrd2gif.py): This program converts an MRD image .h5 file into an animated GIF for quick previews.

### Getting Started
In a command prompt, generate a sample raw dataset:
```
python3 generate_cartesian_shepp_logan_dataset.py -o phantom_raw.h5
```

MRD data is stored in the HDF file format in a hierarchical structure in groups.  The above example creates a ``dataset`` group containing:
```
/dataset/data   Raw k-space data
/dataset/xml    MRD header
```

Start the server in verbose mode by running:
```
python3 main.py -v
```

In another command prompt, start the client and send data to the server for reconstruction:
```
python3 client.py -o phantom_img.h5 phantom_raw.h5
```
The ``-o`` argument specifies the output file and the last argument is the input file.  If the output file already exists, output data is appended to the existing file.

MRD image data are also stored in HDF files arranged by groups. Each run of the client will create a new group with the current date and time.  Images are further grouped by series index, with a sub-group named ``image_x``, where x is image_series_index in the ImageHeader.  For example:
```
/2020-06-26 16:37.157291/image_0/data         Image data
/2020-06-26 16:37.157291/image_0/header       MRD ImageHeader structure
/2020-06-26 16:37.157291/image_0/attributes   MRD MetaAttributes text
```

The [mrd2gif.py](mrd2gif.py) program can be used to convert an MRD Image file into an animated GIF for quick previewing:
```
python3 mrd2gif.py phantom_img.h5
```
A GIF file (animated if multiple images present) is generated in the same folder as the MRD file with the same base file name and the group and sub-groups appended.

The reconstructed images in /tmp/phantom_raw.h5 can be opened in any HDF viewer such as https://www.hdfgroup.org/downloads/hdfview/.  In Python, the [ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools) repository has an interactive ``imageviewer`` tool that can be used to view MRD formatted HDF files.  The syntax is:
```
python3 imageviewer.py phantom_img.h5
```

In MATLAB, the [ISMRMRD](https://github.com/ismrmrd/ismrmrd) library contains helper classes to load and view MRD files.  The files can also be read using MATLAB’s built-in HDF functions:
```
img = h5read('/tmp/phantom_img.h5', '/dataset/2020-06-26 16:37.157291/data');
figure, imagesc(img), axis image, colormap(gray)
```

### Saving incoming data
It may be desirable for the MRD server to save a copy of incoming data from the client.  For example, if the client is an MRI scanner, then the saved data can be used for offline simulations at a later time.  This may be particularly useful when the MRI scanner client is sending image data, as images are not stored in a scanner's raw data file and would otherwise require offline simulation of the MRI scanner reconstruction as well.

The feature may be turned on by starting the server with the ``-s`` option (disabled by default).  Data files are named by the current date/time and stored in ``/tmp/share/saved_data``.  The saved data folder can be changed using the ``-S`` option.  For example, to turn on saving of incoming data in the ``/tmp`` folder, start the server with:
```
python3 main.py -s -S /tmp
```

The ``-s`` flag is enabled in the startup script [start-fire-python-server-with-data-storage.sh](start-fire-python-server-with-data-storage.sh).

Alternatively, this feature can be enabled on a per-session basis when the client calls the server with the config ``savedataonly``.  In this mode, incoming data (raw or image) is saved, but no processing is done and no images are sent back to the client.

The resulting saved data files are in MRD .h5 format and can be used as input for ``client.py`` as detailed above.

### Startup scripts
There are three scripts that may be useful when starting the Python server in a chroot environment (i.e. on the scanner).  When using this server with FIRE, the startup script is specified in the fire.ini file as ``chroot_command``.  The scripts are:

- [start-fire-python-server.sh](start-fire-python-server.sh):  This script takes one optional argument, which is the location of a log file.  If not provided, logging outputs are discarded.

- [sync-code-and-start-fire-python-server.sh](sync-code-and-start-fire-python-server.sh):  This script copies all files from ``/tmp/share/code/`` to ``/opt/code/python-ismrmrd-server/``.  These paths are relative to the chroot container and when run with FIRE, ``/tmp/share/code/`` is a shared folder from the host computer at ``%CustomerIceProgs%\fire\share\code\``.  This "sync" step allows Python code to be modified on the host computer and executed by the Python reconstruction process.  The Python reconstruction program is then started in the same way as in [start-fire-python-server.sh](start-fire-python-server.sh).  This is the startup script specified in the default fire.ini configuration file.  However, this script should not be used in stable projects, as overwriting existing files with those in ``/tmp/share/code`` is likely undesirable.

- [start-fire-python-server-with-data-storage.sh](start-fire-python-server-with-data-storage.sh):  This script is the same as [start-fire-python-server.sh](start-fire-python-server.sh), but saves incoming data (raw k-space readouts or images) to files in ``/tmp/share/saved_data``, which is on the host computer at ``%CustomerIceProgs%\fire\share\saved_data``.  This is useful to save a copy of MRD formatted data (particularly for image-based workflows) for offline development, but this option should be used carefully, as raw data can be quite large and can overwhelm the limited hard drive space on the Windows host.
