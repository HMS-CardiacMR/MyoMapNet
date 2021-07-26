
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
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
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):
                meta = ismrmrd.Meta.deserialize(item.attribute_string)
                slice = extract_minihead_long_param(base64.b64decode(meta['IceMiniHead']).decode('utf-8'), 'AnatomicalPartitionNo')
                item.slice = slice

                # Only process phase images
                if item.image_type is ismrmrd.IMTYPE_PHASE:
                    imgGroup.append(item)
                else:

                    connection.send_image(item)
                    continue

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
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, config, metadata)
            logging.debug("Sending images to client")
            connection.send_image(image)
            imgGroup = []

    finally:
        connection.send_close()

def process_image(images, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Display MetaAttributes for first image
    tmpMeta = ismrmrd.Meta.deserialize(images[0].attribute_string)
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(tmpMeta))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in tmpMeta:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(tmpMeta['IceMiniHead']).decode('utf-8'))

    # Extract some indices for the images
    slice = [img.slice for img in images]
    phase = [img.phase for img in images]

    # Process each group of venc directions separately
    unique_venc_dir = np.unique([ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] for img in images])

    # Start the phase images at series 10.  When interpreted by FIRE, images
    # with the same image_series_index are kept in the same series, but the
    # absolute series number isn't used and can be arbitrary
    last_series = 10
    imagesOut = []
    for venc_dir in unique_venc_dir:
        # data array has dimensions [row col sli phs], i.e. [y x sli phs]
        # info lists has dimensions [sli phs]
        data = np.zeros((images[0].data.shape[2], images[0].data.shape[3], max(slice)+1, max(phase)+1), images[0].data.dtype)
        head = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]
        meta = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]

        for img, sli, phs in zip(images, slice, phase):
            if ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] == venc_dir:
                # print("sli phs", sli, phs)
                data[:,:,sli,phs] = img.data
                head[sli][phs]    = img.getHead()
                meta[sli][phs]    = ismrmrd.Meta.deserialize(img.attribute_string)

        logging.debug("Phase data with venc encoding %s is size %s" % (venc_dir, data.shape,))
        np.save(debugFolder + "/" + "data_" + venc_dir + ".npy", data)

        # Mask out data with high mean temporal diff
        threshold = 250
        data_meandiff = np.mean(np.abs(np.diff(data,3)),3)
        data_masked = data
        data_masked[(data_meandiff > threshold)] = 2048
        np.save(debugFolder + "/" + "data_masked_" + venc_dir + ".npy", data_masked)

        # Normalize and convert to int16
        data_masked = (data_masked.astype(np.float64) - 2048)*32767/2048
        data_masked = np.around(data_masked).astype(np.int16)

        # Re-slice back into 2D images
        for sli in range(data_masked.shape[2]):
            for phs in range(data_masked.shape[3]):
                # Create new MRD instance for the processed image
                # NOTE: from_array() takes input data as [x y z coil], which is
                # different than the internal representation in the "data" field as
                # [coil z y x], so we need to transpose
                tmpImg = ismrmrd.Image.from_array(data_masked[...,sli,phs].transpose())

                # Set the header information
                tmpHead = head[sli][phs]
                tmpHead.data_type          = tmpImg.getHead().data_type
                tmpHead.image_index        = phs + sli*data_masked.shape[3]
                tmpHead.image_series_index = last_series
                tmpImg.setHead(tmpHead)

                # Set ISMRMRD Meta Attributes
                tmpMeta = meta[sli][phs]
                tmpMeta['DataRole']               = 'Image'
                tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
                tmpMeta['WindowCenter']           = '16384'
                tmpMeta['WindowWidth']            = '32768'
                tmpMeta['Keep_image_geometry']    = 1

                # Add image orientation directions to MetaAttributes if not already present
                if tmpMeta.get('ImageRowDir') is None:
                    tmpMeta['ImageRowDir'] = ["{:.18f}".format(tmpHead.read_dir[0]), "{:.18f}".format(tmpHead.read_dir[1]), "{:.18f}".format(tmpHead.read_dir[2])]

                if tmpMeta.get('ImageColumnDir') is None:
                    tmpMeta['ImageColumnDir'] = ["{:.18f}".format(tmpHead.phase_dir[0]), "{:.18f}".format(tmpHead.phase_dir[1]), "{:.18f}".format(tmpHead.phase_dir[2])]

                xml = tmpMeta.serialize()
                logging.debug("Image MetaAttributes: %s", xml)
                tmpImg.attribute_string = xml
                imagesOut.append(tmpImg)

        last_series += 1
    return imagesOut

def extract_minihead_long_param(miniHead, name):
    # Extract a long parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamLong."' + name + r'">{)\s*\d*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    elif res.group(0).isspace():
        return 0
    else:
        return int(res.group(0))
