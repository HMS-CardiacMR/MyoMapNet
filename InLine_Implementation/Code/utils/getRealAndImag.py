import torch


def getRealAndImag(input):
    ndims = input.ndimension()

#     if ndims == 4:
#         return input[:,:,:,0], input[:,:,:,1]
#     if ndims == 5:
#         return input[:,:,:,:,0], input[:,:,:,:,1]
#     if ndims == 6:
#         return input[:,:,:,:,:,0], input[:,:,:,:,:,1]
    
    
    input_real = input.narrow(ndims-1, 0, 1).squeeze(ndims-1)
    input_imag = input.narrow(ndims-1, 1, 1).squeeze(ndims-1)
    
    return input_real, input_imag