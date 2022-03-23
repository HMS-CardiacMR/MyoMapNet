# This is an example of how to intergrate the T1 Mapping program into FIRE.
# This file needs to be moved into the python-ismrmrd-server folder before running.
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
import re
import mrdhelper
import copy
import ctypes

from scipy.optimize import curve_fit

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    # Update this with time to make sure code is updating
    logging.info("---------START NEW LOG 10:13 PM---------")
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

    nfiles = 6
    x_axis = 208
    y_axis = 188
    inversion_times = np.zeros(nfiles)
    pixelDims = (x_axis, y_axis, nfiles)
    pixel_intensities = np.zeros(pixelDims)

    i = 0

    # Continuously parse incoming data parsed from MRD messages
    try:
        for item in connection:
            # logging.info("This is an item: %s\n", item)
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            # Make Sure input is an image and that we only proces the first n images
            # TODO: Find a better way of processing each slices
            if isinstance(item, ismrmrd.Image) and i < nfiles:
                # Capture meta information from the image
                meta = ismrmrd.Meta.deserialize(item.attribute_string)
                # Extract the inverstion time
                inversion_time = extract_minihead_long_param(base64.b64decode(meta['IceMiniHead']).decode('utf-8'), 'TI')
                # Save the inversion time to an array
                # logging.info("Inverstion Time: %d\n", inversion_time)
                inversion_times[i] = inversion_time
                # Extract the pixel information and save the real data
                nparray = item.data
                # logging.info("Data: ", nparray[0,0,...].real)
                pixel_values = nparray[0,0,...].real

                if (pixel_values.shape[0] != 208 or pixel_values.shape[1] != 188):
                    pixel_values = np.rot90(pixel_values, 1) # Rotate 90 degrees
                    pixel_values = np.flipud(pixel_values)   # Flip vertically
                    
                pixel_intensities[:, :, i] = pixel_values
                # Increment Count
                i += 1

                if np.isnan(nparray[0,0,...].real).any() or np.isinf(nparray[0,0,...].real).any():
                    logging.debug("Bad data!!!!!")
                

            #     # Only process phase images
            #     if item.image_type is ismrmrd.IMTYPE_PHASE:
            #         imgGroup.append(item)
            #     else:
            #         connection.send_image(item)
            #         continue

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(inversion_times) > 0:
            logging.info("Processing a group of images (untriggered)")
            t1_params_pre = calculate_T1map(pixel_intensities, inversion_times)
            t1 = make_t1_map(t1_params_pre)

            # Normalize and convert to int16
            data = t1
            # data *= 32767/data.max()
            # data = np.around(data)
            data = data.astype(np.int16)

            logging.info("Setting All Values Above 1800 to 2000")
            # Set all values over 200 to 2000
            data[data > 1800] = 2000

            # Format as ISMRMRD image data
            image = ismrmrd.Image.from_array(data)

            # Set field of view
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

            # Send image back to the client
            logging.debug("Sending images to client")
            connection.send_image(image)

    finally:
        connection.send_close()

def extract_minihead_long_param(miniHead, name):
    # Extract a long parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamDouble."' + name + r'">{)\s*\d*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    elif res.group(0).isspace():
        return 0
    else:
        return int(res.group(0))

def func_orig(x, a, b, c):

    return a*(1-np.exp(-b*x)) + c

def calc_t1value(j, pixel_intensities, inversiontime):

    nx, ny, nti = pixel_intensities.shape
    inversiontime = np.asarray(inversiontime)
    y = np.zeros(nti)
    r = int(j/ny)
    c = int(j%ny)

    p0_initial = [350, 0.005, -150]

    for tino in range(nti):
        y[tino] = pixel_intensities[r,c,tino]

    yf = copy.copy(y)
    sq_err = 100000000.0
    curve_fit_success = False

    for nsignflip in range(3):
        if nsignflip == 0:
            yf[0] = -y[0]
        elif nsignflip == 1:
            yf[0] = -y[0]
            yf[1] = -y[1]
        elif nsignflip == 2:
            yf[0] = -y[0]
            yf[1] = -y[1]
            yf[2] = -y[2]
        try:
            if not np.isnan(yf).any() and not np.isinf(yf).any():
                inversiontime = np.nan_to_num(inversiontime)
                popt,pcov = curve_fit(func_orig, inversiontime, yf, p0=p0_initial)
        except RuntimeError:
            # print("Error - curve_fit failed")
            curve_fit_success = False
            popt = p0_initial

        a1 = popt[0]
        b1 = popt[1]
        c1 = popt[2]

        yf_est = func_orig(inversiontime, a1, b1, c1)
        sq_err_curr = np.sum((yf_est - yf)**2, dtype=np.float32)

        if sq_err_curr < sq_err:
            curve_fit_success = True
            sq_err = sq_err_curr
            a1_opt = a1
            b1_opt = b1
            c1_opt = c1

    if not curve_fit_success:
        a1_opt = 0
        b1_opt = np.finfo(np.float32).max
        c1_opt = 0

    return a1_opt, b1_opt, c1_opt

def calculate_T1map(pixel_intensities, inversiontime):

    nx, ny, nti = pixel_intensities.shape
    t1map = np.zeros([nx, ny, 3])
    pixel_intensities = copy.copy(pixel_intensities)

    if inversiontime[-1] == 0:
        inversiontime = inversiontime[0:-1]
        nTI = inversiontime.shape[0]
        if nti > nTI:
            pixel_intensities = pixel_intensities[:,:,0:nTI]

    for j in range(nx*ny):
        r = int(j / ny)
        c = int(j % ny)
        t1map[r, c, :] = calc_t1value(j, pixel_intensities, inversiontime)

    return t1map

def make_t1_map(t1_params):

    a = t1_params[:, :, 0]
    b = t1_params[:, :, 1]
    c = t1_params[:, :, 2]

    t1 = (1 / b) * (a / (a + c) - 1)

    return t1
