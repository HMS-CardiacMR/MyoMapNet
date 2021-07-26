import ismrmrd
import logging
import numpy as np
import numpy.matlib
import base64
import re
import traceback
import ctypes
import xml.dom.minidom
import xml.etree.ElementTree as ET

from torch import optim
from collections import defaultdict

from myloss import *
from unet import UNet
from utils import *

def process(connection, config, metadata):

    logging.info("---------START NEW MYOMAPNET LOG---------")
    logging.info("Config: %s", config)

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
        logging.info("Improperly formatted metadata: %s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    try:
        t1_weighted_images = defaultdict(list)
        inversion_time_lst = defaultdict(list)
        acquisition_time_lst = defaultdict(list)

        timeScalingFactor = 1000

        # Loop through all the data being sent over
        for item in connection:

            # If the item being sent is an image then process it
            if isinstance(item, ismrmrd.Image):

                # Store header for later use
                oldHeader = item.getHead()

                # Extract slice value
                slice_index = oldHeader.slice

                # Capture meta information from the image
                meta = ismrmrd.Meta.deserialize(item.attribute_string)
                # Extract the inverstion time from ICE Mini Head
                inversion_time = extract_minihead_long_param(base64.b64decode(meta['IceMiniHead']).decode('utf-8'), 'Double', 'TI')
                # Extract the inverstion time from ICE Mini Head
                acquisition_time = float(extract_minihead_long_param(base64.b64decode(meta['IceMiniHead']).decode('utf-8'), 'String', 'AcquisitionTime').strip('"'))

                # Buffer acquisition time
                acquisition_time_lst[slice_index].append(acquisition_time)

                # Store pixel intensity real values
                pixel_intensities = item.data[0,0,...].real

                try:
                    # Normalize the data
                    inversion_time /= timeScalingFactor
                    # Buffer inversion time
                    inversion_time_lst[slice_index].append(inversion_time)
                except:
                    continue

                # Buffer images
                t1_weighted_images[slice_index].append(pixel_intensities)

        ####################################
        #
        # Store Buffered Data To 5D Array
        #
        ####################################

        # Generate numpy array which will be passed into model
        x_length = np.shape(t1_weighted_images[0])[1] # oldHeader.matrix_size[1]
        y_length = np.shape(t1_weighted_images[0])[2] # oldHeader.matrix_size[0]
        nSlices = oldHeader.slice + 1   # Slices is index value so we increment by 1
        nCoils = 1
        images = 8 # 2X number of images (pixel intensities + inversion times)
        pixelDims = (nSlices, nCoils, images, x_length, y_length)
        tst_t1w_TI = np.zeros(pixelDims)

        for slice_index in range(nSlices):

            # Convert list of T1 weighted images into np array
            t1_weighted_IMGs_arr = np.array(t1_weighted_images[slice_index])
            # Store shape for later processing
            arrShape = t1_weighted_IMGs_arr.shape  #[0] = nImages, [1] = X-Axis, [2] = Y-Axis

            # Select only the MOCO images
            t1_weighted_IMGs_Arr_sub6 = t1_weighted_IMGs_arr[6:12,:,:,]

            # Convert the inversion and acquisition times for the 6 MOCO images into an np array
            inversion_time_arr_sub6 = np.array(inversion_time_lst[slice_index][6:12])
            acquisition_time_arr_sub6 = np.array(acquisition_time_lst[slice_index][6:12])

            # Use the acquisition time to sort the inversion times and images
            sort_index_order = np.argsort(acquisition_time_arr_sub6)
            inversion_time_arr_sub6_sorted = inversion_time_arr_sub6[sort_index_order]
            t1_weighted_IMGs_Arr_sub6_sorted = t1_weighted_IMGs_Arr_sub6[sort_index_order]

            # Select only the first 4 images for processing
            inversion_time_arr_sub4 = inversion_time_arr_sub6_sorted[0:4]
            t1_weighted_IMGs_Arr_sub4 = t1_weighted_IMGs_Arr_sub6_sorted[0:4]

            # Reshape 
            inversion_time_arr_sub4 = inversion_time_arr_sub4.reshape(1, inversion_time_arr_sub4.shape[0]) # 1 X N
            inversion_time_Matrix = np.matlib.repmat(inversion_time_arr_sub4, arrShape[1]*arrShape[2], 1)
            inversion_time_Matrix = inversion_time_Matrix.reshape(arrShape[1], arrShape[2], 4)

            # Populate array with inversion times and images
            for iT1Num in range(0,4):
                tst_t1w_TI[slice_index,0,iT1Num,:,:] = t1_weighted_IMGs_Arr_sub4[iT1Num,:,:]
                tst_t1w_TI[slice_index,0,iT1Num+4,:,:] = inversion_time_Matrix[:,:,iT1Num]

        ####################################
        #
        # Normalize Data
        #
        ####################################

        size_tst_t1w_TI = tst_t1w_TI.shape

        for ixSlice in range(0,size_tst_t1w_TI[0]):
            for ixCoil in range(0,size_tst_t1w_TI[1]):
                for ixT1w in range(0,4):
                    max_intensity = np.abs(np.max(t1_weighted_images[ixSlice], axis=0))
                    tmpImg = tst_t1w_TI[ixSlice,ixCoil,ixT1w,:,:]
                    tmpImg = tmpImg/(max_intensity*1.1+np.spacing(1))
                    tst_t1w_TI[ixSlice,ixCoil,ixT1w,:,:] = tmpImg

        # Call myomapnet to produce map
        t1_maps = myomapnetpredict(tst_t1w_TI)

        # Undo scaling factor used to normalize inversion times
        t1_maps *= timeScalingFactor

        # Send T1-Map back to ICE
        for i in range(0, t1_maps.shape[0]):

            # Normalize and convert to int16
            data = t1_maps[i,0,...]
            data = data.astype(np.int16)

            # Apply thresholding to remove outliers
            data[data > 2000] = 2000
            data[data < 75] = 75

            # Transpose the image
            data = np.rot90(data, 1) # Rotate 90 degrees
            data = np.flipud(data)   # Flip vertically

            # Format as ISMRMRD image data
            image = ismrmrd.Image.from_array(data)

            # Set field of view
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

            # Create a copy of the original ISMRMRD Meta attributes and update
            tmpMeta = ismrmrd.Meta.deserialize(image.attribute_string)
            tmpMeta['DataRole']                       = 'Image'
            tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'MYOMAPNET']
            tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'

            # Set colormap values
            tmpMeta['LUTFileName']                    = 'MicroDeltaHotMetal.pal'
            tmpMeta['WindowCenter']                   = '1300'
            tmpMeta['WindowWidth']                    = '1300'

            metaXml = tmpMeta.serialize()

            logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
            logging.debug("Image data has %d elements", image.data.size)

            image.attribute_string = metaXml

            # Send image back to the client
            logging.debug("Sending images to client")
            connection.send_image(image)

    finally:
        connection.send_close()

def extract_minihead_long_param(string, data_type, var_name):

    r = re.compile(fr'<Param{data_type}\."{var_name}">' r'{ "?(.+)"? }')
    res = r.search(string)

    if res is None:
        return None
    elif res.group(1).isspace():
        return 0
    elif data_type == 'Double':
        return int(res.group(1))
    else:
        return res.group(1)

def multiply_elems(x):
    m = 1
    for e in x:
        m *= e
    return m

def myomapnetpredict(tst_t1w_TI):

    params = Parameters()

    ####################################
    #
    # Create Model
    #
    ####################################

    net = UNet(8, 1)

    # Sets the module in evaluation mode.
    net.eval()

    if params.multi_GPU:
        net = torch.nn.DataParallel(net, device_ids=params.device_ids[:-1]).cuda()
    else:
        net = torch.nn.DataParallel(net, device_ids=params.device_ids[:-1])
        net.to(params.device)

    # Dynamicically build path to models directory
    params.model_save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    try:
        ###########################################
        #
        # INITIALIZATIONS
        #
        ############################################

        optimizer = optim.SGD(net.parameters(), lr=params.args.lr, momentum=0.8)

        ###########################################
        #
        # LOAD LATEST (or SPECIFIC) MODEL
        #
        ############################################

        models = os.listdir(params.model_save_dir)
        models = [m for m in models if m.endswith('.pth')]
        s_epoch = 2196  ## -1: load latest model or start from 1 if there is no saved models
                        ##  0: don't load any model; start from model #1
                        ##  num: load model #num

        def load_model(epoch):
            print('loading model at epoch ' + str(epoch))
            print('Total Models:', len(models))
            model = torch.load(os.path.join(params.model_save_dir, models[0][0:11] + str(epoch) + '.pth'), map_location=torch.device('cpu'))
            print('Using Model:', os.path.join(params.model_save_dir, models[0][0:11] + str(epoch) + '.pth'))
            net.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimizer'])

        if s_epoch == -1:
            if len(models) == 0:
                s_epoch = 1
            else:
                s_epoch = max([int(epo[11:-4]) for epo in models[:]])
                load_model(s_epoch)
        elif s_epoch == 0:
            s_epoch = 1
        else:
            try:
                load_model(s_epoch)
            except:
                print('Model {0} does not exist!'.format(s_epoch))

        try:
            tst_N = tst_t1w_TI.shape[0]
            tst_lst = list(range(0,tst_N))
            batchSize = 3

            with torch.no_grad():
                for idx in range(0, tst_N, batchSize):
                    try:
                        X = Variable(torch.FloatTensor(tst_t1w_TI[tst_lst[idx:idx + batchSize]])).to('cpu')

                        xs = X.shape

                        X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))

                        y_pred = net(X.to('cpu')).to('cpu')

                        pred_T1_5 = y_pred.reshape((xs[0],1,xs[3],xs[4]))

                        t1_maps = pred_T1_5.cpu().data.numpy()

                        return t1_maps

                    except Exception as e:
                        traceback.print_exc()
                        continue
        except Exception as e:
            traceback.print_exc()
            print(e)

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
