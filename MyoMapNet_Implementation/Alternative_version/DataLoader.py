import h5py
import numpy as np
from sys import exit


def load_training_data(path):
    """
    loading training dataset
    
    Argument:
    path -- string, the path to the training dataset
    
    Returns:
    x_train -- numpy array, contains a 4D array for number of images, number of channels (4 T1W and 4 IT), width and heigh of each image
    y_train -- numpy array, contains a 3D array for number of images, width and heigh of each image (T1 map)
    
    """
    
    training_data = h5py.File(path)
    x_train = np.array(training_data.get("allT1wTIs"))
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[4])


    x_train = x_train.transpose(3, 2, 0, 1)

    y_train = np.array(training_data.get("allT1Maps"))
    y_train = y_train.transpose(2, 0, 1)
    y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1], y_train.shape[2])

    return x_train, y_train

def load_validation_data(path):
    """
    loading validation dataset
    
    Argument:
    path -- string, the path to the validation dataset
    
    Returns:
    validation_data -- numpy array, contains a 4D array for number of images, number of channels (4 T1W and 4 IT), width and heigh of each image
    y_validation -- numpy array, contains a 3D array for number of images, width and heigh of each image
    bpMask -- numpy array, contains the ROI for blood pool
    myoMask -- numpy array, contains the ROI for myocardium
    myobpMask -- numpy array, contains the ROI for myocardium and blood pool
    
    """
    
    validation_data = h5py.File(path)
    x_validation = np.array(validation_data.get("allT1wTIs"))
    x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2],
                                        x_validation.shape[4])

    x_validation = x_validation.transpose(3, 2, 0, 1)
    y_validation = np.array(validation_data.get("allT1Maps"))
    y_validation = y_validation.transpose(2, 0, 1)
    y_validation = y_validation.reshape(y_validation.shape[0], 1, y_validation.shape[1], y_validation.shape[2])

    bpMask = np.array(validation_data.get("bpMask"))
    bpMask = bpMask.transpose(2, 0, 1)
    bpMask = bpMask.reshape(bpMask.shape[0], 1, bpMask.shape[1], bpMask.shape[2])

    myoMask = np.array(validation_data.get("myoMask"))
    myoMask = myoMask.transpose(2, 0, 1)
    myoMask = myoMask.reshape(myoMask.shape[0], 1, myoMask.shape[1], myoMask.shape[2])

    myobpMask = np.array(validation_data.get("myobpMask"))
    myobpMask = myobpMask.transpose(2, 0, 1)
    myobpMask = myobpMask.reshape(myobpMask.shape[0], 1, myobpMask.shape[1], myobpMask.shape[2])

    return x_validation, y_validation, bpMask, myoMask, myobpMask

