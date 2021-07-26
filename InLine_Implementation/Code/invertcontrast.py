
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    acqGroup = []
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)):
                    acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, config, metadata)
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, config, metadata)
                    connection.send_image(image)
                    imgGroup = []

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, config, metadata)
            connection.send_image(image)
            acqGroup = []

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, config, metadata)
            connection.send_image(image)
            imgGroup = []

    finally:
        connection.send_close()


def process_raw(group, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha PE RO phs] array
    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    phs = [acquisition.idx.phase                for acquisition in group]

    # Use the zero-padded matrix size
    data = np.zeros((group[0].data.shape[0], 
                     metadata.encoding[0].encodedSpace.matrixSize.y, 
                     metadata.encoding[0].encodedSpace.matrixSize.x, 
                     max(phs)+1), 
                    group[0].data.dtype)

    rawHead = [None]*(max(phs)+1)

    for acq, lin, phs in zip(group, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            # TODO: Account for asymmetric echo in a better way
            data[:,lin,-acq.data.shape[1]:,phs] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead[phs] is None) or (np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])):
                rawHead[phs] = acq.getHead()

    # Flip matrix in RO/PE to be consistent with ICE
    data = np.flip(data, (1, 2))

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
    # Data will be [PE RO phs]
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.x)/2)
    data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.x]

    # Remove phase oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.y)/2)
    data = data[offset:offset+metadata.encoding[0].reconSpace.matrixSize.y,:]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Format as ISMRMRD image data
    imagesOut = []
    for phs in range(data.shape[2]):
        # Create new MRD instance for the processed image
        # NOTE: from_array() takes input data as [x y z coil], which is
        # different than the internal representation in the "data" field as
        # [coil z y x], so we need to transpose
        tmpImg = ismrmrd.Image.from_array(data[...,phs].transpose())

        # Set the header information
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[phs]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = phs

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'
        tmpMeta['Keep_image_geometry']    = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    # Call process_image() to invert image contrast
    imagesOut = process_image(imagesOut, config, metadata)

    return imagesOut


def process_image(images, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to the more intuitive [x y z cha img]
    data = data.transpose()

    # Reformat data again to [y x z cha img], i.e. [row col] for the first two
    # dimensions.  Note we will need to undo this later prior to sending back
    # to the client
    data = data.transpose((1, 0, 2, 3, 4))

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    # Normalize and convert to int16
    data = data.astype(np.float64)
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Invert image contrast
    data = 32767-data
    data = np.abs(data)
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgInverted.npy", data)

    # Reformat data from [row col z cha img] back to [x y z cha img] before sending back to client
    data = data.transpose((1, 0, 2, 3, 4))

    currentSeries = 0

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):
        # Create new MRD instance for the inverted image
        # NOTE: from_array() takes input data as [x y z coil], which is
        # different than the internal representation in the "data" field as
        # [coil z y x].  However, we already transposed this data when
        # extracting it earlier.
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg])
        data_type = imagesOut[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type

        # Unused example, as images are grouped by series before being passed into this function now
        # oldHeader.image_series_index = currentSeries

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'INVERT']
        tmpMeta['WindowCenter']                   = '16384'
        tmpMeta['WindowWidth']                    = '32768'
        tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'
        tmpMeta['Keep_image_geometry']            = 1
        # tmpMeta['ROI_example']                    = create_example_roi(data.shape)

        # Example for setting colormap
        # tmpMeta['LUTFileName']            = 'MicroDeltaHotMetal.pal'

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    return imagesOut

# Create an example ROI <3
def create_example_roi(img_size):
    t = np.linspace(0, 2*np.pi)
    x = 16*np.power(np.sin(t), 3)
    y = -13*np.cos(t) + 5*np.cos(2*t) + 2*np.cos(3*t) + np.cos(4*t)

    # Place ROI in bottom right of image, offset and scaled to 10% of the image size
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    y = (y-np.min(y)) / (np.max(y) - np.min(y))
    x = (x * 0.08*img_size[0]) + 0.82*img_size[0]
    y = (y * 0.10*img_size[1]) + 0.80*img_size[1]

    rgb = (1,0,0)  # Red, green, blue color -- normalized to 1
    thickness  = 1 # Line thickness
    style      = 0 # Line style (0 = solid, 1 = dashed)
    visibility = 1 # Line visibility (0 = false, 1 = true)

    roi = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)
    return roi
