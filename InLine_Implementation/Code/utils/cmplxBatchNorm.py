import torch.nn as nn
import torch
import numpy as np
from utils.polarTransforms import * 


# def normalizeMagnitudeBatch(x):
#     ''' normalize each slice alone'''
#     ndim = x.ndimension()
#     if ndim == 3:
#         xs = x.reshape((x.shape[0],x.shape[1]*x.shape[2]))
#     elif ndim==4:
#         xs = x.reshape((x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
#     
#     return (x - torch.mean(xs, ndim-2, keepdim=True).unsqueeze(ndim-1))/torch.std(xs, ndim-2, keepdim=True).unsqueeze(ndim-1)


def magnitude(input):
    real, imag = torch.unbind(input, -1)
    return (real ** 2 + imag ** 2) ** 0.5

    
def complexSTD(x):
    ''' 
    Standard deviation of real and imaginary channels
    STD = sqrt( E{(x-mu)(x-mu)*} ), where * is the complex conjugate, 
        
    - Source: https://en.wikipedia.org/wiki/Variance#Generalizations
    '''
    mu = torch.mean(torch.mean(x, 2, True), 3, True)
    
    xm = torch.sum(((x-mu)**2), 2, True); #(a+ib)(a-ib)* = a^2 + b^2
    return torch.mean(torch.mean(xm, 2, True), 3, True)**(0.5)

def normalizeComplexBatch(x):
    ''' normalize real and imaginary channels'''
    return (x - torch.mean(torch.mean(x, 2, True), 3, True))/complexSTD(x)

def log_mag(x, polar=False):
    if not polar:
        x = cylindricalToPolarConversion(x)

    ndims = x.ndimension()
    mag, phase = torch.unbind(x, -1)
    x = torch.stack([torch.log(1 + mag), phase], dim=ndims-1)

    if not polar:
        x = polarToCylindricalConversion(x)

    return x

def exp_mag(x, polar=False):
    if not polar:
        x = cylindricalToPolarConversion(x)

    ndims = x.ndimension()
    mag, phase = torch.unbind(x, -1)
    x = torch.stack([torch.exp(mag)-1, phase], dim=ndims-1)

    if not polar:
        x = polarToCylindricalConversion(x)

    return x

def mult_list(x, l):
    m = 1
    for i in x[l:]:
        m *= i
    return m


def normalizeComplexBatch_byMagnitudeOnly(x, polar=False, normalize_over_channel=False):
    ''' normalize the complex batch by making the magnitude of mean 1 and std 1, and keep the phase as it is'''
    ndims = x.ndimension()
    shift_mean = 1
    if not polar:
        x = cylindricalToPolarConversion(x)

    mag, phase = torch.unbind(x, -1)
    mdims = mag.ndimension()


    if normalize_over_channel:## might not work for large tensors: cuda runtime error (9) : invalid configuration
        if ndims == 5:
            mag_shaped = mag.reshape((mag.shape[0], mag.shape[1] * mag.shape[2] * mag.shape[3]))
            normalized_mag = (mag - torch.mean(mag_shaped, mdims - 3, keepdim=True).unsqueeze(mdims - 2).unsqueeze(mdims - 1)) / torch.std(
                mag_shaped, mdims - 3, keepdim=True).unsqueeze(mdims - 2).unsqueeze(mdims - 1) + shift_mean

        elif ndims == 6:
            mag_shaped = mag.reshape((mag.shape[0], mag.shape[1] * mag.shape[2] * mag.shape[3] * mag.shape[4]))
            normalized_mag = (mag - torch.mean(mag_shaped, mdims - 4, keepdim=True).unsqueeze(mdims - 3).unsqueeze(mdims - 2).unsqueeze(
                mdims - 1)) / torch.std(mag_shaped, mdims - 4, keepdim=True).unsqueeze(mdims - 3).unsqueeze(mdims - 2).unsqueeze(
                mdims - 1) + shift_mean
    else:
        if ndims == 5:
            mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2] * mag.shape[3]))
            normalized_mag = (mag - torch.mean(mag_shaped, mdims - 2, keepdim=True).unsqueeze(mdims - 1)) / torch.std(
                mag_shaped, mdims - 2, keepdim=True).unsqueeze(mdims - 1) + shift_mean

        elif ndims == 6:
            mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2] * mag.shape[3] * mag.shape[4]))
            normalized_mag = (mag - torch.mean(mag_shaped, mdims - 3, keepdim=True).unsqueeze(mdims - 2).unsqueeze(
                mdims - 1)) / torch.std(mag_shaped, mdims - 3, keepdim=True).unsqueeze(mdims - 2).unsqueeze(
                mdims - 1) + shift_mean

    x = torch.stack([normalized_mag, phase], dim=ndims-1)
    #     x[x.ne(x)] = 0.0
    if not polar:
        x = polarToCylindricalConversion(x)
    #     print('normalizeComplexBatch_byMagnitudeOnly', np.isnan(x.data.cpu()).sum())

    return x
    

class ComplexBatchNormalize(nn.Module):
    
    def __init__(self):
        super(ComplexBatchNormalize, self).__init__()
    
    def forward(self, input):
        return normalizeComplexBatch_byMagnitudeOnly(input)
#         return normalizeComplexBatch(input)
    
    
    