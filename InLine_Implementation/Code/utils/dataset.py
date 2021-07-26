import torch
from torch.utils import data
from parameters import Parameters
from scipy.io import loadmat, savemat
import numpy as np
import os
from saveNet import *
# params = Parameters()

def resizeImage(img,newSize, Interpolation=False):
    
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    
    if Interpolation:
        return imresize(img, tuple(newSize), interp='bilinear')
    else:
        
        x1 = (img.shape[0]-newSize[0])//2
        x2 = img.shape[0]-newSize[0] - x1

        y1 = (img.shape[1]-newSize[1])//2
        y2 = img.shape[1]-newSize[1] - y1

        if img.ndim == 3:
            if x1 > 0:
                img = img[x1:-x2,:,:]
            elif x1 < 0:
                img = np.pad(img, ((-x1, -x2),(0,0),(0,0)), 'constant') #((top, bottom), (left, right))
    
            if y1 > 0:
                img = img[:,y1:-y2,:]
            elif y1 < 0:
                img = np.pad(img, ((0,0),(-y1, -y2),(0,0)), 'constant') #((top, bottom), (left, right))
        
        elif img.ndim ==4:
            if x1 > 0:
                img = img[x1:-x2,:,:,:]
            elif x1 < 0:
                img = np.pad(img, ((-x1, -x2),(0,0),(0,0),(0,0)), 'constant') #((top, bottom), (left, right))
    
            if y1 > 0:
                img = img[:,y1:-y2,:,:]
            elif y1 < 0:
                img = np.pad(img, ((0,0),(-y1, -y2),(0,0),(0,0)), 'constant') #((top, bottom), (left, right))
        return img.squeeze()

def resize3DVolume(data,newSize, Interpolation=False):
    
    ndim = data.ndim
    if ndim < 3:
        return None
    elif ndim == 3:
        data = np.expand_dims(data, 3)
    
    if Interpolation:
        return imresize(data, tuple(newSize), interp='bilinear')
    
    elif ndim == 4:    
        x1 = (data.shape[0]-newSize[0])//2
        x2 = data.shape[0]-newSize[0] - x1

        y1 = (data.shape[1]-newSize[1])//2
        y2 = data.shape[1]-newSize[1] - y1

        z1 = (data.shape[2]-newSize[2])//2
        z2 = data.shape[2]-newSize[2] - z1

        if x1 > 0:
            data = data[x1:-x2,:,:,:]
        elif x1 < 0:
            data = np.pad(data, ((-x1, -x2),(0,0),(0,0),(0,0)), 'constant') #((top, bottom), (left, right))

        if y1 > 0:
            data = data[:,y1:-y2,:,:]
        elif y1 < 0:
            data = np.pad(data, ((0,0),(-y1, -y2),(0,0),(0,0)), 'constant') #((top, bottom), (left, right))

        if z1 > 0:
            data = data[:,:,z1:-z2,:]
        elif z1 < 0:
            data = np.pad(data, ((0,0),(0,0),(-z1, -z2),(0,0)), 'constant') #((top, bottom), (left, right))

        return data.squeeze()

def getPatientSlicesURLs(patient_url):
        islices = list()
        oslices = list()
        for fs in os.listdir(patient_url+'/InputData/Input_realAndImag/'):
            islices.append(patient_url+'/InputData/Input_realAndImag/' + fs)
            
        for fs in os.listdir(patient_url+'/CSRecon/CSRecon_Data_small/'):
            oslices.append(patient_url+'/CSRecon/CSRecon_Data_small/' + fs)
        islices = sorted(islices,key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))
        oslices = sorted(oslices,key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))
        
        return (islices, oslices)


def getDatasetGenerators(params):
    
    
    num_slices_per_patient = []
    params.input_slices = []
    params.groundTruth_slices = []
    params.us_rates = [];
    num_slices_in_slice_patient = [];
    
    P = loadmat(params.net_save_dir+'lgePatients.mat')['lgePatients']
    pNames = [i[0][0] for i in P]
    usRates = [i[1][0] for i in P]
    
    k=-1
    for p in pNames:
        k += 1
        for dir in params.dir:
            pdir = dir + p
            if os.path.exists(pdir):        
                params.patients.append(pdir)
                slices = getPatientSlicesURLs(pdir)
                num_slices_per_patient.append(len(slices[0]))
                num_slices_in_slice_patient += [len(slices[0])]*len(slices[0])
                params.input_slices = np.concatenate((params.input_slices, slices[0]))
                params.groundTruth_slices = np.concatenate((params.groundTruth_slices, slices[1]))
                params.us_rates = np.concatenate([params.us_rates, usRates[k]*np.ones(len(slices[0]))])
                continue
    
    print('-- Number of Datasets: '+str(len(params.patients)))
    
    params.num_slices_per_patient = num_slices_per_patient
                    
    training_ptns = round(params.training_percent * len(num_slices_per_patient))
    
    training_end_indx = sum(num_slices_per_patient[0:training_ptns+1])
    evaluation_end_indx = training_end_indx + sum(num_slices_per_patient)
    
    params.training_patients_index = range(0,training_ptns+1)

    dim = params.img_size[:]
    dim.append(2)
    
    tr_samples = 2
    
    
    if params.network_type== '2D':
        training_DS = DataGenerator(input_IDs=params.input_slices[:training_end_indx:tr_samples], 
                                    output_IDs=params.groundTruth_slices[:training_end_indx:tr_samples],
                                    undersampling_rates = params.us_rates[:training_end_indx:tr_samples],
                                    dim=dim, 
                                    n_channels=params.n_channels,
                                    complex_net=params.complex_net,
                                    nums_slices = num_slices_in_slice_patient[:training_end_indx:tr_samples])

        validation_DS = DataGenerator(input_IDs=params.input_slices[training_end_indx:evaluation_end_indx], 
                                      output_IDs=params.groundTruth_slices[training_end_indx:evaluation_end_indx],
                                      undersampling_rates = params.us_rates[training_end_indx:evaluation_end_indx],
                                      dim=dim, 
                                      n_channels=params.n_channels,
                                      complex_net=params.complex_net,
                                      nums_slices = num_slices_in_slice_patient[training_end_indx:evaluation_end_indx])

    elif params.network_type== '3D':
        if params.num_slices_3D > 50: #whole volume Feeding
            training_DS = DataGenerator(input_IDs=params.patients[:training_ptns], 
                                        output_IDs=params.patients[:training_ptns],
                                        undersampling_rates = params.us_rates[:training_end_indx:tr_samples],
                                        dim=dim, 
                                        n_channels=params.n_channels,
                                        complex_net=params.complex_net,
                                        nums_slices = num_slices_in_slice_patient[:training_end_indx:tr_samples])
            
        else: #moving window feeding
            training_DS = DataGenerator(input_IDs=params.input_slices[:training_end_indx:tr_samples], 
                                        output_IDs=params.groundTruth_slices[:training_end_indx:tr_samples],
                                        undersampling_rates = params.us_rates[:training_end_indx:tr_samples],
                                        dim=dim, 
                                        n_channels=params.n_channels,
                                        complex_net=params.complex_net,
                                        nums_slices = num_slices_in_slice_patient[:training_end_indx:tr_samples])



        validation_DS = DataGenerator(input_IDs=params.patients[training_ptns:], 
                                      output_IDs=params.patients[training_ptns:],
                                      undersampling_rates = params.us_rates[training_end_indx:evaluation_end_indx],
                                      dim=[256, 256, 200,2], 
                                      n_channels=params.n_channels,
                                      complex_net=params.complex_net,
                                      nums_slices = num_slices_in_slice_patient[training_end_indx:evaluation_end_indx],
                                      mode='testing')
    

    training_DL = data.DataLoader(training_DS, batch_size=params.batch_size, shuffle=True, num_workers=params.data_loders_num_workers)    
#     validation_DL = data.DataLoader(validation_DS, batch_size=params.batch_size, shuffle=False, num_workers=params.data_loders_num_workers)        
    validation_DL = data.DataLoader(validation_DS, batch_size=1, shuffle=False, num_workers=params.data_loders_num_workers)        
    
    return training_DL, validation_DL, params  

def get_moving_window(indx, num_sl, total_num_sl):
    if indx-num_sl//2 < 1:
        return range(1, num_sl+1)
    
    if indx+num_sl//2 > total_num_sl:
        return range(total_num_sl-num_sl+1 ,total_num_sl+1)

    return range(indx-num_sl//2, indx+num_sl//2+1)


class DataGenerator(data.Dataset):
    'Generates data for Keras'
    def __init__(self, input_IDs, output_IDs, undersampling_rates=None, dim=(256,256,2), n_channels=1,complex_net=True ,nums_slices=None, mode='training'):
        'Initialization'
        self.dim = dim
        self.output_IDs = output_IDs
        self.input_IDs = input_IDs
        self.n_channels = n_channels
        self.undersampling_rates = undersampling_rates
        self.nums_slices = nums_slices
        self.complex_net = complex_net
        self.mode = mode
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.input_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'
        
        X, y, orig_size = self.__data_generation(index, self.input_IDs[index], self.output_IDs[index])
        if self.undersampling_rates is not None:
            usr = self.undersampling_rates[index]
        else:
            usr = None
            
        return X, y, self.input_IDs[index], orig_size, usr


    def __data_generation(self, index ,input_IDs_temp, output_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        if self.complex_net:
            if len(self.dim)== 3:
                return self.generate_data2D(index, input_IDs_temp, output_IDs_temp)
            elif len(self.dim) > 3 and self.mode == 'training':
                if params.num_slices_3D > 50: #whole volume Feeding
                    return self.generate_data3D(index, input_IDs_temp, output_IDs_temp)
                else: #moving window feeding
                    return self.generate_data3D_moving_window(index, input_IDs_temp, output_IDs_temp)
            elif len(self.dim) > 3 and self.mode == 'testing':
                return self.generate_data3D_testing(index, input_IDs_temp, output_IDs_temp)
        else:
            if len(self.dim)==2:
                return self.generate_data2D(index, input_IDs_temp, output_IDs_temp)
            else:
                return self.generate_data3D(index, input_IDs_temp, output_IDs_temp)
            
    
    def generate_data2D(self, index, input_IDs_temp, output_IDs_temp):
        # Initialization
        X = np.zeros((self.n_channels, *self.dim))
        y = np.zeros((self.n_channels, *self.dim))

        # Generate data
        img = loadmat(input_IDs_temp)['Input_realAndImag']
        orig_size = [img.shape[0], img.shape[1]]
#         for i, ID in enumerate(input_IDs_temp):
        X[0,] = resizeImage(img,[self.dim[0],self.dim[1]])

#         for i, ID in enumerate(output_IDs_temp):
        y[0,:,:,0] = resizeImage(loadmat(output_IDs_temp)['Data'],[self.dim[0],self.dim[1]])
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        return X, y, orig_size

    def generate_data3D(self, index, patients, out_patients):
        '''
        Read 3D volumes or stack of 2D slices
        '''
        Stack_2D = True
        
        if Stack_2D:
            slices = getPatientSlicesURLs(patients)
            X = np.zeros((1, self.dim[0], self.dim[1], self.dim[2], 2))
            y = np.zeros((1, self.dim[0], self.dim[1], self.dim[2], 2))
            
            z1 = (len(slices[0])-self.dim[2])//2
            z2 = len(slices[0])-self.dim[2] - z1
            
            sz=0
            if z1 > 0:
                rng = range(z1, len(slices[0])-z2)
                sz = -z1
            elif z1 < 0:
                rng = range(0, len(slices[0]))
                sz = z1
            elif z1 == 0:
                rng = range(0, self.dim[2])
            
            for sl in rng:
                img = loadmat(slices[0][sl])['Input_realAndImag']
                orig_size = [img.shape[0], img.shape[1]]
                try:
                    X[0,:,:,sl+sz,:] = resizeImage(img,[self.dim[0],self.dim[1]])
                    
                    y[0,:,:,sl+sz,0] = resizeImage(loadmat(slices[1][sl])['Data'],[self.dim[0],self.dim[1]])
                except:
                    stop = 1
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            return X, y, orig_size
        else:
            pass

    def generate_data3D_moving_window(self, index, input_IDs_temp, output_IDs_temp):
        '''
            Moving window
        '''
        # Initialization
        X = np.zeros((self.n_channels, *self.dim))
        y = np.zeros((self.n_channels, *self.dim))

        sl_indx = int(input_IDs_temp.split('/')[-1][8:-4])
        
        rng = get_moving_window(sl_indx, self.dim[2], self.nums_slices[index])
        
        i = 0
        # Generate data
#         print(input_IDs_temp)
#         print('sl_indx->', sl_indx, '  nslices->',self.dim[2], 'max_slices', self.nums_slices[index] )
        for sl in  rng:
#             print(sl)
            in_sl_url = '/'.join(input_IDs_temp.split('/')[0:-1])+'/Input_sl' + str(sl) + '.mat'
            out_sl_url = '/'.join(output_IDs_temp.split('/')[0:-1])+'/Input_sl' + str(sl) + '.mat'
            try:            
                img = loadmat(in_sl_url)['Input_realAndImag']
            except:
                print('Data Loading Error ..... !')
            orig_size = [img.shape[0], img.shape[1]]
            X[0,:,:,i,:] = resizeImage(img,[self.dim[0],self.dim[1]])
            y[0,:,:,i,0] = resizeImage(loadmat(out_sl_url)['Data'],[self.dim[0],self.dim[1]])
            i += 1
#         print('---------------------------------')    
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        return X, y, orig_size

    def generate_data3D_testing(self, index, patients, out_patients):
        def ceildiv(a, b):
            return -(-a // b)
        slices = getPatientSlicesURLs(patients)
        X = np.zeros((1, self.dim[0], self.dim[1], len(slices[0]), 2))
        y = np.zeros((1, self.dim[0], self.dim[1], len(slices[0]), 2))
        
        for sl in range(0, len(slices[0])):
            img = loadmat(slices[0][sl])['Input_realAndImag']
            orig_size = [img.shape[0], img.shape[1]]
            X[0,:,:,sl,:] = resizeImage(img,[self.dim[0],self.dim[1]])
            
            y[0,:,:,sl,0] = resizeImage(loadmat(slices[1][sl])['Data'],[self.dim[0],self.dim[1]])
            
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        
        
#         n_batchs = ceildiv(len(slices[0]), self.dim[2])
# 
#         # Initialization        
#         X = np.zeros((n_batchs,1, *self.dim))
#         y = np.zeros((n_batchs,1, *self.dim))
# 
#         ds_sl = 0
#         for bt in range(0, n_batchs):
#             for sl in range(0, self.dim[2]):
#                 if ds_sl >= len(slices[0]):
#                     break    
# #                 print('ds_sl:',ds_sl, 'sl:',sl, 'bt:', bt)
#                 img = loadmat(slices[0][ds_sl])['Input_realAndImag']
#                 orig_size = [img.shape[0], img.shape[1]]
#                 X[bt,0,:,:,sl,:] = resizeImage(img,[self.dim[0],self.dim[1]])
#                 
#                 y[bt,0,:,:,sl,0] = resizeImage(loadmat(slices[1][ds_sl])['Data'],[self.dim[0],self.dim[1]])
#                 X = np.nan_to_num(X)
#                 y = np.nan_to_num(y)
#                 ds_sl += 1
        return X, y, orig_size
        
        

        
        
# class DataGenerator(data.Dataset):
#     'Generates data for Keras'
#     def __init__(self, input_IDs, output_IDs, undersampling_rates=None, dim=(256,256,2), n_channels=1,complex_net=True ,nums_slices=None):
#         'Initialization'
#         self.dim = dim
#         self.output_IDs = output_IDs
#         self.input_IDs = input_IDs
#         self.n_channels = n_channels
#         self.undersampling_rates = undersampling_rates
#         self.nums_slices = nums_slices
#         self.complex_net = complex_net
#         
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.input_IDs)
# 
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         if len(self.dim)==2 or (len(self.dim) ==3 and self.complex_net):
#             return self.getItem2D(index)
#         else:
#             return self.getItem3D(index)
#         
#     def getItem2D(self, index):    
#         # Generate data
#         X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
#         if self.undersampling_rates is not None:
#             usr = self.undersampling_rates[index]
#         else:
#             usr = None
#             
#         return X, y, self.input_IDs[index], orig_size, usr
# 
#     def getItem3D(self, index):    
#         # Generate data
#         X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
#         if self.undersampling_rates is not None:
#             usr = self.undersampling_rates[index]
#         else:
#             usr = None
#             
#         return X, y, self.input_IDs[index], orig_size, usr
# 
# 
#     def __data_generation(self, input_IDs_temp, output_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.zeros((self.n_channels, *self.dim))
#         y = np.zeros((self.n_channels, *self.dim))
# 
#         # Generate data
#         img = loadmat(input_IDs_temp)['Input_realAndImag']
#         orig_size = [img.shape[0], img.shape[1]]
# #         for i, ID in enumerate(input_IDs_temp):
#         X[0,] = resizeImage(img,[self.dim[0],self.dim[1]])
# 
# #         for i, ID in enumerate(output_IDs_temp):
#         y[0,:,:,0] = resizeImage(loadmat(output_IDs_temp)['Data'],[self.dim[0],self.dim[1]])
#         X = np.nan_to_num(X)
#         y = np.nan_to_num(y)
#         return X, y, orig_size
    

    
    