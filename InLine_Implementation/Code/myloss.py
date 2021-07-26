
#
# myloss.py : implementation of the Dice coeff and the associated loss
#

from math import exp

import torch
from torch.autograd import Function, Variable
from torch.nn.modules.loss import _Loss

import numpy as np
import torch.nn.functional as F
from saveNet import *


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input, target) + 0.0001
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2*self.inter.float()/self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i+1)


class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - dice_coeff(F.sigmoid(input), target)

def gaussian_nd(shape, sigma=1.0, mu=0.0):
    """ create a n dimensional gaussian kernel for the given shape """
    m = np.meshgrid(*[np.linspace(-1,1,s) for s in shape])
    d = np.sqrt(np.sum([x*x for x in m], axis=0))
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g / np.sum(g)

def create_NDwindow(window_shape,channel):
    _nD_window = torch.Tensor(gaussian_nd(window_shape))
    window = Variable(_nD_window.expand(channel, 1, *window_shape).contiguous())
    return window



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window



def _ssim(img1, img2, window, window_size, channel, size_average=True, full=False):
    padd = 0

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
    
        channel = img1.shape[1]

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)




def _ssim_3d(img1, img2, window, window_size, channel, size_average=True, full=False):
    padd = 0

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIM_3D(torch.nn.Module):
    def __init__(self, window_shape=(11,11,11), size_average=True, channels=1):
        super(SSIM_3D, self).__init__()
        self.window_shape = window_shape
        self.size_average = size_average
        self.channels = channels
        self.window = create_NDwindow(window_shape, self.channels)

    def forward(self, img1, img2):
        
        self.window.to(img1.get_device())
        self.window  = self.window.type_as(img1)

        return _ssim_3d(img1, img2, self.window, self.window_shape, self.channels, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True, full=False):
    (_, channel, height, width) = img1.size()

    real_size = min(window_size, height, width)
    window = create_window(real_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, real_size, channel, size_average, full=full)


def msssim(img1, img2, window_size=11, size_average=True):
    # TODO: fix NAN results
    if img1.size() != img2.size():
        raise RuntimeError('Input images must have the same shape (%s vs. %s).' %
                           (img1.size(), img2.size()))
    if len(img1.size()) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           len(img1.size()))

    if type(img1) is not Variable or type(img2) is not Variable:
        raise RuntimeError('Input images must be Variables, not %s' % 
                            img1.__class__.__name__)

    weights = Variable(torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
    if img1.is_cuda:
        weights = weights.cuda(img1.get_device())

    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.cat(mssim)
    mcs = torch.cat(mcs)
    return (torch.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
            (mssim[levels-1] ** weights[levels-1]))


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def kspace_consistency(img1, img2):
    k1 = torch.fft(img1, signal_ndim=2, normalized=False)
    k2 = torch.fft(img2, signal_ndim=2, normalized=False)
    e = torch.ifft((k2-k1), signal_ndim=2, normalized=False)
    return e.abs_().sum()

class EdgeLoss(_Loss):
    def __init__(self, beta=0.2):
        super(EdgeLoss, self).__init__()
        self.beta = beta
        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        
        self.mseCriterion = torch.nn.modules.MSELoss()
        
    def forward(self, img, ref):
        
        self.gaussian_filter_horizontal = self.gaussian_filter_horizontal.to(ref.device)
        self.gaussian_filter_vertical = self.gaussian_filter_vertical.to(ref.device)
        self.sobel_filter_horizontal = self.sobel_filter_horizontal.to(ref.device)
        self.sobel_filter_vertical = self.sobel_filter_vertical.to(ref.device)
        
        blur_horizontal = self.gaussian_filter_horizontal(ref)
        blurred_img = self.gaussian_filter_vertical(blur_horizontal)
        
        grad_x = self.sobel_filter_horizontal(blurred_img)
        grad_y = self.sobel_filter_vertical(blurred_img)
        
        # COMPUTE THICK EDGES
        
        edge = torch.sqrt(grad_x**2 + grad_y**2)
        
        loss = (1-self.beta)*self.mseCriterion(img, ref) + self.beta * self.mseCriterion(edge*img, edge*ref)
        return loss

class EdgeLoss_3D(_Loss):
    def __init__(self, beta=0.9):
        super(EdgeLoss_3D, self).__init__()
        self.beta = beta
        filter_size = 5
        generated_filters = gaussian(filter_size,1.0)

        self.gaussian_filter_x = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(filter_size,1,1), padding=(filter_size//2,0,0))
        self.gaussian_filter_x.weight.data.copy_(generated_filters.reshape([filter_size,1,1]))
        self.gaussian_filter_x.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        
        self.gaussian_filter_y = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,filter_size,1), padding=(0,filter_size//2,0))
        self.gaussian_filter_y.weight.data.copy_(generated_filters.reshape([1,filter_size,1]))
        self.gaussian_filter_y.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.gaussian_filter_z = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,1,filter_size), padding=(0,0,filter_size//2))
        self.gaussian_filter_z.weight.data.copy_(generated_filters.reshape([1,1,filter_size]))
        self.gaussian_filter_z.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # Ref: http://www.aravind.ca/cs788h_Final_Project/gradient_estimators.htm#idsb
        sobel_filter = np.array([[[-1, -3, -1],
                                  [-3, -6, -3],
                                  [-1, -3, -1]],
                                 
                                 [[ 0,  0,  0],
                                  [ 0,  0,  0],
                                  [ 0,  0,  0]],
                                 
                                 [[ 1,  3,  1],
                                  [ 3,  6,  3],
                                  [ 1,  3,  1]]])
        

#         sobel_filter = np.array([[1, 0, -1],
#                                  [2, 0, -2],
#                                  [1, 0, -1]])

        self.sobel_filter_x = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_x.weight.data.copy_(torch.from_numpy(sobel_filter.transpose((2,0,1))))
        self.sobel_filter_x.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.sobel_filter_y = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[1]//2)
        self.sobel_filter_y.weight.data.copy_(torch.from_numpy(sobel_filter.transpose((1,0,2))))
        self.sobel_filter_y.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.sobel_filter_z = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[2]//2)
        self.sobel_filter_z.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_z.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        
        self.mseCriterion = torch.nn.modules.loss.MSELoss()#torch.nn.modules.MSELoss()
        
    def forward(self, img, ref):
        
        self.gaussian_filter_x = self.gaussian_filter_x.to(ref.device)
        self.gaussian_filter_y = self.gaussian_filter_y.to(ref.device)
        self.gaussian_filter_z = self.gaussian_filter_z.to(ref.device)
        
        self.sobel_filter_x = self.sobel_filter_x.to(ref.device)
        self.sobel_filter_y = self.sobel_filter_y.to(ref.device)
        self.sobel_filter_z = self.sobel_filter_z.to(ref.device)
        
        blur_x = self.gaussian_filter_x(ref)
        blur_y = self.gaussian_filter_y(blur_x)
        blurred_img = self.gaussian_filter_z(blur_y)
        
        grad_x = self.sobel_filter_x(blurred_img)
        grad_y = self.sobel_filter_y(blurred_img)
        grad_z = self.sobel_filter_z(blurred_img)

        # COMPUTE THICK EDGES
        
        edge = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        edge = normalizeBatch_torch(edge)
#         dim = edge.shape
#         max_edg, _ = torch.max(edge.reshape(dim[0], dim[1], dim[2]*dim[3]*dim[4]), 2, True)
#         max_edg = max_edg.unsqueeze(-1).unsqueeze(-1).expand_as(edge)
#         
#         min_edg, _ = torch.min(edge.reshape(dim[0], dim[1], dim[2]*dim[3]*dim[4]), 2, True)
#         min_edg = min_edg.unsqueeze(-1).unsqueeze(-1).expand_as(edge)
#         
#         edge = (edge - min_edg) / max_edg
        
#         saveTensorToMat(edge, 'edge1')
#         saveTensorToMat(ref, 'ref1')
#         saveTensorToMat(img, 'img1')
#         saveTensorToMat(edge*ref, 'ed_ref')
#         saveTensorToMat(edge*img, 'ed_img')
        
        loss = (1-self.beta)*self.mseCriterion(img, ref) + self.beta * ((edge*img - edge*ref)**2).mean()
        return loss

def normalizeBatch_torch(p):
    ''' normalize each slice alone'''
    if torch.std(p) == 0:
        raise ZeroDivisionError
    shape = p.shape
    if p.ndimension() == 4:
        pv = p.reshape([shape[0],shape[1],shape[2]*shape[3]])
        mean =  pv.mean(dim=2, keepdim=True).unsqueeze(p.ndimension()-1)
        std =  pv.std(dim=2, keepdim=True).unsqueeze(p.ndimension()-1)
    elif p.ndimension() == 5:
        pv = p.reshape([shape[0],shape[1],shape[2]*shape[3]*shape[4]])
        mean =  pv.mean(dim=2, keepdim=True).unsqueeze(p.ndimension()-2).unsqueeze(p.ndimension()-1)
        std =  pv.std(dim=2, keepdim=True).unsqueeze(p.ndimension()-2).unsqueeze(p.ndimension()-1)
       
    return (p - mean)/std


class ExponentialLoss(_Loss):
    def __init__(self):
        super(ExponentialLoss, self).__init__()
#         self.register_parameter('beta', torch.nn.Parameter(torch.Tensor([0.001]),requires_grad = True) )
        self.mseCriterion = torch.nn.modules.MSELoss()
#         self.beta.fill_(0.001)
            
    def forward(self, img, ref):
#         self.beta = self.beta.to(img.device)
#         print(self.beta)
        
        
#         return self.mseCriterion(self.beta * torch.exp(img) , self.beta * torch.exp(ref))
        return self.mseCriterion(img, ref) + 0.005*self.mseCriterion(torch.exp(img) , torch.exp(ref))



class KspaceConsistency(_Loss):

    def forward(self, img1, img2):
        return kspace_consistency(img1, img2)

    
class TotalVariations(_Loss):

    def forward(self, img1):
        return torch.sum(torch.abs(img1[:, :, :-1] - img1[:, :, 1:])) + torch.sum(torch.abs(img1[:, :-1, :] - img1[:, 1:, :]))


class weighted_mse(_Loss):
    def __init__(self):
        super(weighted_mse, self).__init__()

    def forward(self, input, output, weight):
        return torch.sum(weight * (input - output) ** 2) / input.numel()

class weighted_mae(_Loss):
    def __init__(self):
        super(weighted_mae, self).__init__()
    def forward(self, input, output, weight):
        return torch.sum(weight * torch.abs(input - output)) / input.numel()

