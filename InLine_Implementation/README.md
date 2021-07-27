# MyoMapNet FIRE Integration

The inline integration is implemented using the Siemens Framework for Image Reconstruction (FIRE) prototype framework. Briefly, the FIRE framework provides an interface for raw data or image data between the Siemens Image Reconstruction Environment (ICE) pipeline and an external environment like Python. The pre-trained MyoMapNet4, Pre+PostGd model was deployed in a containerized (chroot) Python 3.6 environment compatible with the FIRE framework.

## Prerequisite

The MyoMapNet program was implemented using Python `3.6.13` and `Pip 18.1`

All the necessary packages are listed in the requirments.txt file located in the Code directory.

## Create Virtual Enviornment

It is recommended to first create a virtual enviornment prior to running the code.

Create Python venv with Python 3.6:

     python3.6 -m venv myomapnet-venv

Activate venv:

    source myomapnet-venv/bin/activate

Install dependinces:

    pip install -r requirments.txt

## Launching FIRE Server

Once the necessary libraries have been installed the FIRE server can simply be launched like so:

    python main.py

You will then see the following message appear in your console:

    2021-07-26 18:22:16,465 - Starting server and listening for data at 0.0.0.0:9002

This indicates the FIRE is running and will process any data sent to it via port 9002.