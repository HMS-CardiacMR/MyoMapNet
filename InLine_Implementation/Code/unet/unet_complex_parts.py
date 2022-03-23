'''
Created on May 21, 2018

@author: helrewaidy
'''
# sub-parts of the U-Net model

import torch

import torch.nn as nn
import torch.nn.functional as F
import utils.cmplxBatchNorm as cmplxBatchNorm
from complexnet.cmplxconv import ComplexConv2d, ComplexConv3d, ComplexConvTranspose2d
from complexnet.cmplxbn import ComplexBatchNormalize
from complexnet.radialbn2 import RadialBatchNorm2d, RadialBatchNorm3d
from complexnet.cmplxupsample import ComplexUpsample
from complexnet.cmplxdropout import ComplexDropout2d
from complexnet.cmplxmodrelu import ModReLU
from complexnet.kafactivation import KAF2D
from complexnet.zrelu import ZReLU
from utils.fftutils import fftshift2d, ifftshift2d

from parameters import Parameters

params = Parameters()


def Activation(*args):
    if params.activation_func == 'CReLU':
        return nn.ReLU(inplace=True)
    elif params.activation_func == 'CLeakyeak':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif params.activation_func == 'modReLU':
        return ModReLU(*args)
    elif params.activation_func == 'KAF2D':
        return KAF2D(*args)
    elif params.activation_func == 'ZReLU':
        return ZReLU(polar=False)


if params.network_type == '2D':
    ComplexConv = ComplexConv2d
    RadialBatchNorm = RadialBatchNorm2d
elif params.network_type == '3D':
    ComplexConv = ComplexConv3d
    RadialBatchNorm = RadialBatchNorm3d


class double_conv(nn.Module):
    '''(conv => ReLU => BN) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            # #             nn.Conv3d(in_ch, out_ch, [3, 3, 1], padding=(1,1,0)),
            #             ComplexBatchNormalize(in_ch),
            # #             cmplxBatchNorm.ComplexBatchNormalize(),
            ComplexConv(in_ch, out_ch, 3, padding=1),
            RadialBatchNorm(out_ch),  # ComplexBatchNormalize(out_ch),
            Activation(out_ch),  # nn.ReLU(inplace=True),
            # #             cmplxBatchNorm.ComplexBatchNormalize(),
            #             nn.BatchNorm3d(out_ch, affine=False), #, affine=False
            # ComplexDropout2d(params.dropout_ratio),
            ComplexConv(out_ch, out_ch, 3, padding=1),
            RadialBatchNorm(out_ch),  # ComplexBatchNormalize(out_ch)
            Activation(out_ch)  # nn.ReLU(inplace=True),
            # #             cmplxBatchNorm.ComplexBatchNormalize(),
            # #             nn.BatchNorm3d(out_ch, affine=False), #, affine=False
            # ComplexDropout2d(params.dropout_ratio)

        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):
    def __init__(self, in_ch):
        super(down_conv, self).__init__()
        down_stride = (2, 2, 1) if params.network_type == '3D' else (2, 2)
        self.conv = nn.Sequential(
            ComplexConv(in_ch, in_ch, 3, stride=down_stride, padding=1),
            RadialBatchNorm(in_ch),  # ComplexBatchNormalize(in_ch),
            Activation(in_ch)  # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class bnorm(nn.Module):
    def __init__(self, in_ch):
        super(bnorm, self).__init__()
        self.norm = RadialBatchNorm(in_ch)
    def forward(self, x):
        x = self.norm(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.down_conv = down_conv(in_ch)
        self.double_conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        down_x = self.down_conv(x)
        x = self.double_conv(down_x)
        return x, down_x


class bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, residual_connection=True):
        super(bottleneck, self).__init__()
        self.residual_connection = residual_connection
        self.down_conv = down_conv(in_ch)
        self.double_conv = nn.Sequential(
            ComplexDropout2d(params.dropout_ratio),
            ComplexConv(in_ch, 2 * in_ch, 3, padding=1),
            RadialBatchNorm(2 * in_ch),  # ComplexBatchNormalize(2*in_ch),
            Activation(2 * in_ch),  # nn.ReLU(inplace=True),
            # #             cmplxBatchNorm.ComplexBatchNormalize(),
            #             nn.BatchNorm3d(out_ch, affine=False), #, affine=False
            ComplexDropout2d(params.dropout_ratio),
            ComplexConv(2 * in_ch, out_ch, 3, padding=1),
            RadialBatchNorm(out_ch),  # ComplexBatchNormalize(out_ch),
            # #             cmplxBatchNorm.ComplexBatchNormalize(),
            #             nn.BatchNorm3d(out_ch, affine=False), #, affine=False
            Activation(out_ch),  # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down_x = self.down_conv(x)
        if self.residual_connection:
            x = self.double_conv(down_x) + down_x
        else:
            x = self.double_conv(down_x)

        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            #             self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            #             self.up = nn.Upsample(scale_factor=(2,2,1), mode='trilinear')
            upsample_mode = 'trilinear' if params.network_type == '3D' else 'bilinear'
            upsample_scale_factor = (2, 2, 1) if params.network_type == '3D' else (2, 2)
            self.up = ComplexUpsample(scale_factor=upsample_scale_factor, mode=upsample_mode)
        else:
            self.up = ComplexConvTranspose2d(in_ch, in_ch, (2, 2), stride=(2, 2))

        self.conv = nn.Sequential(
            #             nn.Conv3d(in_ch, out_ch, [3, 3, 1], padding=(1,1,0)),
            ComplexConv(in_ch * 2, in_ch, 3, padding=1),
            RadialBatchNorm(in_ch),  # ComplexBatchNormalize(in_ch),
            # #                     cmplxBatchNorm.ComplexBatchNormalize(),
            #             nn.BatchNorm3d(out_ch, affine=False), #, affine=False
            Activation(in_ch),  # nn.ReLU(inplace=True),
            ComplexConv(in_ch, out_ch, 3, padding=1),
            RadialBatchNorm(out_ch),  # ComplexBatchNormalize(out_ch),
            # #                     cmplxBatchNorm.ComplexBatchNormalize(),
            #             nn.BatchNorm3d(out_ch, affine=False), #, affine=False
            Activation(out_ch)  # nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class mag_phase_combine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(mag_phase_combine, self).__init__()
        self.conv1d = nn.Sequential(
            # HOSSAM :: Please Remove the patch Normalization from here if it didn't work
            #             nn.BatchNorm2d(in_ch),
            ComplexConv(in_ch, out_ch, 1, padding=(0, 0))
        )

    def forward(self, x):
        t = torch.split(x, int(x.size()[2] / 2), dim=2)
        xt = [i for i in t]
        x1 = xt[0]
        x2 = xt[1]
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1d(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = ComplexConv(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=True, groups=1, apply_BN=False, apply_activation=False):
        super(conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.apply_BN = apply_BN
        layers = []
        layers.append(ComplexConv(in_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation, bias=bias, groups=groups))

        if apply_BN:
            layers.append(RadialBatchNorm(out_ch))

        if apply_activation:
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        return self.conv(x)


class conv_3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=True, groups=1, apply_BN=False, apply_activation=False):
        super(conv_3D, self).__init__()
        # self.padding = dilation * (kernel_size - 1) // 2
        self.padding = [dilation * (ks - 1) // 2 for ks in kernel_size]

        self.apply_BN = apply_BN
        layers = []
        layers.append(ComplexConv3d(in_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation, bias=bias, groups=groups))

        if apply_BN:
            layers.append(RadialBatchNorm3d(out_ch))

        if apply_activation:
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        return self.conv(x)


class conv_ri(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=True, groups=1, apply_BN=False, apply_activation=False):
        super(conv_ri, self).__init__()
        self.padding = dilation * (kernel_size[0] - 1) // 2
        self.apply_BN = apply_BN
        layers = []
        layers.append(nn.Conv3d(in_ch, out_ch, kernel_size, padding=(self.padding, self.padding, 0), dilation=dilation, bias=bias, groups=groups))

        if apply_BN:
            layers.append(nn.BatchNorm3d(out_ch))

        if apply_activation:
            layers.append(nn.ReLU(out_ch))

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        return self.conv(x)


class mixed_dilations_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=(2, 3), apply_BN=False, kspace=False):
        super(mixed_dilations_block, self).__init__()
        self.apply_BN = apply_BN
        self.kspace = kspace

        self.conv1 = conv(in_ch, out_ch//2, kernel_size=kernel_size, dilation=1, apply_BN=apply_BN, kspace=kspace)
        self.conv1_d2 = conv(in_ch, out_ch - out_ch//2, kernel_size=kernel_size, dilation=dilations[0], apply_BN=apply_BN, kspace=kspace)

        self.outconv = conv(out_ch, out_ch, kernel_size=kernel_size, dilation=1, apply_BN=apply_BN, kspace=kspace)

    def forward(self, x):

        x1_1 = self.conv1(x)
        x1_2 = self.conv1_d2(x)

        return self.outconv(torch.cat([x1_1, x1_2], 1))


class mixed_dilations_block_1(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=(2, 3), apply_BN=False, kspace=False):
        super(mixed_dilations_block_1, self).__init__()
        self.apply_BN = apply_BN
        self.kspace = kspace

        self.inconv = conv(in_ch, out_ch // 3, kernel_size=kernel_size, dilation=1, apply_BN=apply_BN, kspace=kspace)

        self.conv1 = conv(in_ch, out_ch//2, kernel_size=kernel_size, dilation=1, apply_BN=apply_BN, kspace=kspace)
        self.conv1_d2 = conv(in_ch, out_ch - out_ch//2, kernel_size=kernel_size, dilation=dilations[0], apply_BN=apply_BN, kspace=kspace)

        self.conv2 = conv(out_ch, out_ch//3, kernel_size=kernel_size, dilation=1, apply_BN=apply_BN, kspace=kspace)
        self.conv2_d3 = conv(out_ch, out_ch-2*(out_ch//3), kernel_size=kernel_size, dilation=dilations[1], apply_BN=apply_BN, kspace=kspace)

        self.conv3 = conv(out_ch, out_ch, kernel_size=kernel_size, dilation=1, apply_BN=apply_BN, kspace=kspace)



    def forward(self, x):

        x0_1 = self.inconv(x)
        x1_1 = self.conv1(x)
        x1_2 = self.conv1_d2(x)

        x2 = torch.cat([x1_1, x1_2], 1)

        x2_1 = self.conv2(x2)
        x2_3 = self.conv2_d3(x2)

        x3 = torch.cat([x0_1, x2_1, x2_3], 1)

        return self.conv3(x3)


from complexnet.frqweighting import FrequencyWeighting2d

class stacked_convs_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=1, apply_BN=False, apply_activation=False):
        super(stacked_convs_block, self).__init__()
        # s = params.img_size
        # grd = np.meshgrid(range(0, s[0]), range(0, s[1]))
        # self.ksp_weight = FrequencyWeighting2d(grid=grd, center=(s[0]//2, s[1]//2), sigma=(1, 1), polar=False)
        self.conv_stack = nn.Sequential(
            conv(in_ch, in_ch, kernel_size=kernel_size, dilation=dilations, apply_BN=apply_BN, apply_activation=False),
            conv(in_ch, in_ch, kernel_size=kernel_size+2, dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv(in_ch, in_ch, kernel_size=kernel_size+2, dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv(in_ch, out_ch, kernel_size=kernel_size, dilation=dilations, apply_BN=False, apply_activation=False)
        )

    def forward(self, x):
        return self.conv_stack(x)


class stacked_3Dconvs_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3,3), dilations=1, apply_BN=False, apply_activation=False, bottelneck_ratio=1):
        super(stacked_3Dconvs_block, self).__init__()
        self.conv_stack = nn.Sequential(
            conv_3D(in_ch, np.round(bottelneck_ratio * in_ch), kernel_size=kernel_size, dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv_3D(np.round(bottelneck_ratio * in_ch), np.round(bottelneck_ratio * in_ch), kernel_size=kernel_size, dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            # conv_3D(out_ch, out_ch, kernel_size=kernel_size, dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv_3D(np.round(bottelneck_ratio * in_ch), out_ch, kernel_size=kernel_size, dilation=dilations, apply_BN=False, apply_activation=False)
        )

    def forward(self, x):
        return self.conv_stack(x)


class CUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CUNet, self).__init__()
        self.conv_3D_1 = conv_3D(in_ch, 32, kernel_size=(3,3,3), dilation=1, apply_BN=False, apply_activation=True)
        self.conv_3D_2 = conv_3D(32, in_ch,
                               kernel_size=(3, 3, 3), dilation=1, apply_BN=False, apply_activation=True)

        self.inc = inconv(in_ch * params.moving_window_size, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.bottleneck = bottleneck(128, 128, False)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.up4 = up(32, 32)
        self.ouc = outconv(32, out_ch)

    def forward(self, x):
        x = self.conv_3D_1(x)
        x = self.conv_3D_2(x)

        shape = x.shape
        x = x.reshape((shape[0], shape[1] * shape[2], shape[3], shape[4], shape[5]))
        x1 = self.inc(x)
        x2, down_x1 = self.down1(x1)
        x3, down_x2 = self.down2(x2)
        x4 = self.bottleneck(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # + x0
        x = self.ouc(x)
        return x

class stacked_convs_block_ri(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=1, apply_BN=False, apply_activation=False):
        super(stacked_convs_block_ri, self).__init__()
        # s = params.img_size
        # grd = np.meshgrid(range(0, s[0]), range(0, s[1]))
        # self.ksp_weight = FrequencyWeighting2d(grid=grd, center=(s[0]//2, s[1]//2), sigma=(1, 1), polar=False)
        self.conv_stack = nn.Sequential(
            conv_ri(in_ch, in_ch, kernel_size=(3,3,1), dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv_ri(in_ch, in_ch, kernel_size=(5,5,1), dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv_ri(in_ch, in_ch, kernel_size=(5,5,1), dilation=dilations, apply_BN=apply_BN, apply_activation=apply_activation),
            conv_ri(in_ch, out_ch, kernel_size=(3,3,1), dilation=dilations, apply_BN=False, apply_activation=False)
        )

    def forward(self, x):
        return self.conv_stack(x)


class kspace_image_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=(2, 3), apply_BN=False):
        super(kspace_image_block, self).__init__()

        # self.mdb_k1 = mixed_dilations_block(in_ch, in_ch, kernel_size=kernel_size, dilations=dilations,
        #                                     apply_BN=apply_BN, kspace=True)
        #
        # self.mdb_i1 = mixed_dilations_block_1(in_ch, out_ch, kernel_size=kernel_size, dilations=dilations,
        #                                       apply_BN=apply_BN, kspace=False)
        #
        # # self.mdb_i2 = mixed_dilations_block_1(32, out_ch, kernel_size=3, dilations=(2, 3),
        # #                                       apply_BN=False, kspace=False)

        self.stc_k1 = stacked_convs_block(in_ch, in_ch, kernel_size=kernel_size, dilations=dilations,
                                            apply_BN=apply_BN, kspace=True)

        self.stc_i1 = stacked_convs_block(in_ch, out_ch, kernel_size=kernel_size, dilations=dilations,
                                              apply_BN=apply_BN, kspace=False)

    def forward(self, x):
        # x = self.mdb_k1(x)
        # x = fftshift2d(torch.ifft(x, 2, normalized=True), [2, 3])
        # x = self.mdb_i1(x)
        # # x = self.mdb_i2(x)

        x = self.stc_k1(x) #+ x
        x = torch.ifft(x, 2, normalized=False) #fftshift2d(torch.ifft(x, 2, normalized=True), [2, 3])
        x = self.stc_i1(x) + x

        return x


class kspace_image_block_2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=(2, 3), apply_BN=False):
        super(kspace_image_block_2, self).__init__()

        self.sc_k1 = mixed_dilations_block(in_ch, in_ch, kernel_size=kernel_size, dilations=dilations,
                                            apply_BN=apply_BN, kspace=True)

        self.mdb_i1 = mixed_dilations_block_1(in_ch, out_ch, kernel_size=kernel_size, dilations=dilations,
                                              apply_BN=apply_BN, kspace=False)
        # self.mdb_i2 = mixed_dilations_block_1(32, out_ch, kernel_size=3, dilations=(2, 3),
        #                                       apply_BN=False, kspace=False)

    def forward(self, x):
        x = self.mdb_k1(x)
        x = fftshift2d(torch.ifft(x, 2, normalized=True), [2, 3])
        x = self.mdb_i1(x)
        # x = self.mdb_i2(x)

        return x




import matplotlib.pyplot as plt
import numpy as np

def tensorshow(x, sl_dims=(0, 0), range=[0, 0.001]):
    plt.figure()
    plt.imshow(np.sqrt(x.cpu().data.numpy()[sl_dims[0], sl_dims[1], :, :, 0] ** 2 +
                       x.cpu().data.numpy()[sl_dims[0], sl_dims[1], :, :, 1] ** 2),
               cmap='gray', vmin=range[0], vmax=range[1])
    plt.show()

# class mixed_dilated_conv(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, apply_BN = False  ):
#         super(mixed_dilated_conv, self).__init__()
#         self.padding = dilation*(kernel_size-1)//2
#
#         self.conv1 = conv(in_ch, out_ch, kernel_size, dilation=1, apply_BN)
#         self.conv1_d = conv(in_ch, out_ch, kernel_size, dilation=dilation, apply_BN)
#
#         self.conv2 = conv(in_ch, out_ch, kernel_size, dilation, apply_BN)
#         self.conv3 = conv(in_ch, out_ch, kernel_size, dilation, apply_BN)
#
#
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
