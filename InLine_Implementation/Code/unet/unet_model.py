#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import time

import torch

from parameters import Parameters
from saveNet import *
import torch.nn as nn
import torch.nn.functional as F
from complexnet.cmplxfc import ComplexLinear
import matplotlib.pyplot as plt
import numpy as np
from complexnet.gridkernels import GriddingKernels, GaussianGriddingKernels
from utils.fftutils import fftshift2d, ifftshift2d
params = Parameters()
from utils.data_vis import tensorshow, ntensorshow
from utils.cmplxBatchNorm import magnitude, normalizeComplexBatch_byMagnitudeOnly, log_mag, exp_mag

import unet.unet_complex_parts as unet_cmplx
import unet.unet_realImag_parts as unet_realImag
import unet.unet_real_parts as unet_real


def get_kspace_bands(num_bands=4, margins=[]):
    # if len(margins) == 0:
    #     margins = range(3, num_bands*2+1, 2)
    num_bands = 4
    margins = [0, 3, 5, 7, 9, 0]
    s = params.img_size[0]
    max_r = s * 1.4142 #np.sqrt(2)
    rads = [0, max_r/30, max_r/10, max_r/6, max_r]
    bands = np.zeros((num_bands, s, s))
    marg_bands = np.zeros((num_bands, s, s))
    center = [s//2, s//2]
    Y, X = np.ogrid[:s, :s]
    dist = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    for b in range(1, num_bands+1):
        marg_bands[b-1, :, :] = ( (dist >= (rads[b-1] - margins[b-1])) * (dist <= (rads[b] + margins[b]))).astype(int)
        bands[b - 1, :, :] = ((dist >= rads[b - 1]) * (dist <= rads[b])).astype(int)

        # plt.figure(b)
        # plt.imshow(bands[b-1, :, :])
        # plt.show()
    return marg_bands, bands

def tensorshow(x, sl_dims=(0, 0), range=[0, 0.001]):
    plt.figure()
    plt.imshow(np.sqrt(x.cpu().data.numpy()[sl_dims[0], sl_dims[1], :, :, 0] ** 2 +
                       x.cpu().data.numpy()[sl_dims[0], sl_dims[1], :, :, 1] ** 2),
               cmap='gray', vmin=range[0], vmax=range[1])
    plt.show()


#########################################################
# MODEL = 0 # Original U-net implementation
# MODEL = 1 # Shallow U-net implementation with combination of magnitude and phase
# MODEL = 2 # The OLD working Real and Imaginary network (Residual Network with one global connection) REAL and Imaginary channels through the network
# MODEL = 3 # Shallow U-Net with residual connection
# MODEL = 4 # Complex Shallow U-Net with residual connection
# MODEL = 5 # Complex Shallow U-Net with residual connection and 32-coil input
# MODEL = 6 # Complex fully connected layer
#MODEL == 9:  # Complex Network takes neighborhood matrix input and image domain output
##########################################################

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size=[64, 64]):
        super(UNet, self).__init__()
        if params.MODEL == 0:  # Deep Unet
            self.inc = unet_real.inconv(n_channels, 64)
            self.down1 = unet_real.down(64, 128)
            self.down2 = unet_real.down(128, 256)
            self.down3 = unet_real.down(256, 512)
            self.down4 = unet_real.down(512, 512)
            self.up1 = unet_real.up(1024, 256)
            self.up2 = unet_real.up(512, 128)
            self.up3 = unet_real.up(256, 64)
            self.up4 = unet_real.up(128, 64)
            self.outc = unet_real.outconv(64, n_classes)

        elif params.MODEL == 1:  # Shallow Unet
            #         self.inc = inconv(n_channels, 64)
            self.inc = unet_real.inconv(n_channels, 64)
            self.down1 = unet_real.down(64, 128)
            #         self.down2 = down(128, 256)
            self.down2 = unet_real.down(128, 128)
            self.down3 = unet_real.down(256, 512)
            self.down4 = unet_real.down(512, 512)
            self.up1 = unet_real.up(1024, 256)
            self.up2 = unet_real.up(512, 128)
            #         self.up3 = up(256, 64)
            self.up3 = unet_real.up(256, 128)
            #         self.up4 = up(128, 64)
            self.up4 = unet_real.up(192, 64)
            self.outc = unet_real.outconv(64, 1)
            self.mpcomb = unet_real.mag_phase_combine(2, n_classes)

        elif params.MODEL == 2:  # The OLD working Real and Imaginary network (based on 3D convolutions) (Residual Network with one global connection)
            self.inc = unet_realImag.inconv(n_channels, 64)
            self.down1 = unet_realImag.down(64, 128)
            self.down2 = unet_realImag.down(128, 128)
            #             self.down3 = down(256, 512)
            #             self.down4 = down(512, 512)
            #             self.up1 = up(1024, 256)
            #             self.up2 = up(512, 128)
            self.up3 = unet_realImag.up(256, 128)
            self.up4 = unet_realImag.up(192, 64)
            self.outc = unet_realImag.outconv(64, n_channels)
        #             self.mpcomb = mag_phase_combine(2, n_classes)
        elif params.MODEL == 3:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.down1 = unet_cmplx.down(64, 128)
            self.down2 = unet_cmplx.down(128, 256)
            self.bottleneck = unet_cmplx.bottleneck(256, 256, False)
            self.up2 = unet_cmplx.up(256, 128)
            self.up3 = unet_cmplx.up(128, 64)
            self.up4 = unet_cmplx.up(64, 64)
            self.ouc = unet_cmplx.outconv(64, n_channels)

        elif params.MODEL == 3.1:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.conv64 = unet_cmplx.inconv(64, 64)
            self.conv64_32 = unet_cmplx.inconv(64, 32)
            self.conv32_64 = unet_cmplx.inconv(32, 64)
            self.conv32 = unet_cmplx.inconv(32, 32)
            self.outc = unet_cmplx.outconv(64, n_channels)
        #             self.ouc = unet_cmplx.outconv(n_channels, n_channels)
        elif params.MODEL == 3.2:
            self.inc = unet_cmplx.inconv(n_channels, 256)
            self.down1 = unet_cmplx.down(256, 128)
            self.down2 = unet_cmplx.down(128, 64)
            self.bottleneck = unet_cmplx.bottleneck(64, 64, False)
            self.up2 = unet_cmplx.up(64, 128)
            self.up3 = unet_cmplx.up(128, 256)
            self.up4 = unet_cmplx.up(256, n_channels)
        #             self.ouc = unet_cmplx.outconv(n_channels, n_channels)
        elif params.MODEL == 3.3:

            s = 20
            self.num_GPUs = 3
            self.multiGPU = True
            DataParallel = True

            if not self.multiGPU:
                self.conv1 = unet_cmplx.conv(n_channels, s)

                self.conv2 = unet_cmplx.conv(s, s)
                self.conv2_d2 = unet_cmplx.conv(s, s, dilation=2)

                self.conv3 = unet_cmplx.conv(2 * s, s)
                self.conv3_d2 = unet_cmplx.conv(2 * s, s, dilation=2)

                self.conv4 = unet_cmplx.conv(2 * s, s)
                self.conv4_d3 = unet_cmplx.conv(2 * s, s, dilation=3)

                self.conv5 = unet_cmplx.conv(2 * s, s)
                self.conv5_d3 = unet_cmplx.conv(2 * s, s, dilation=3)

                self.conv6 = unet_cmplx.conv(2 * s, s)
                self.out_conv = unet_cmplx.outconv(s, 1)
            #                 if DataParallel:
            #                     self = torch.nn.DataParallel(self,device_ids=params.device_ids).cuda()

            else:
                if self.num_GPUs == 3:
                    GPUs = [[0, 5], [1, 4], [2, 3]]  # [[0,1,2],[5,4,3]] #[5,7,6]
                    self.conv1 = unet_cmplx.conv(n_channels, s).cuda(GPUs[0][0])

                    self.conv2 = unet_cmplx.conv(s, s).cuda(GPUs[0][0])
                    self.conv2_d2 = unet_cmplx.conv(s, s, dilation=2).cuda(GPUs[0][0])

                    self.conv3 = unet_cmplx.conv(2 * s, s).cuda(GPUs[0][0])
                    self.conv3_d2 = unet_cmplx.conv(2 * s, s, dilation=2).cuda(GPUs[1][0])

                    self.conv4 = unet_cmplx.conv(2 * s, s).cuda(GPUs[1][0])
                    self.conv4_d3 = unet_cmplx.conv(2 * s, s, dilation=3).cuda(GPUs[1][0])

                    self.conv5 = unet_cmplx.conv(2 * s, s).cuda(GPUs[2][0])
                    self.conv5_d3 = unet_cmplx.conv(2 * s, s, dilation=3).cuda(GPUs[2][0])

                    self.conv6 = unet_cmplx.conv(2 * s, s).cuda(GPUs[2][0])
                    self.out_conv = unet_cmplx.outconv(s, 1).cuda(GPUs[0][0])

                    if DataParallel:
                        self.conv1 = torch.nn.DataParallel(self.conv1, device_ids=GPUs[0])
                        self.conv2 = torch.nn.DataParallel(self.conv2, device_ids=GPUs[0])
                        self.conv2_d2 = torch.nn.DataParallel(self.conv2_d2, device_ids=GPUs[0])
                        self.conv3 = torch.nn.DataParallel(self.conv3, device_ids=GPUs[0])
                        self.conv3_d2 = torch.nn.DataParallel(self.conv3_d2, device_ids=GPUs[1])
                        self.conv4 = torch.nn.DataParallel(self.conv4, device_ids=GPUs[1])
                        self.conv4_d3 = torch.nn.DataParallel(self.conv4_d3, device_ids=GPUs[1])
                        self.conv5 = torch.nn.DataParallel(self.conv5, device_ids=GPUs[2])
                        self.conv5_d3 = torch.nn.DataParallel(self.conv5_d3, device_ids=GPUs[2])
                        self.conv6 = torch.nn.DataParallel(self.conv6, device_ids=GPUs[2])
                        self.out_conv = torch.nn.DataParallel(self.out_conv, device_ids=GPUs[0])


                elif self.num_GPUs == 4:
                    GPUs = [[0, 5], [1, 4], [2, 7], [3, 6]]  # [[0,1,2,3],[5,4,7,6]]
                    self.conv1 = unet_cmplx.conv(n_channels, s).cuda(GPUs[0][0])

                    self.conv2 = unet_cmplx.conv(s, s).cuda(GPUs[0][0])
                    self.conv2_d2 = unet_cmplx.conv(s, s, dilation=2).cuda(GPUs[0][0])

                    self.conv3 = unet_cmplx.conv(2 * s, s).cuda(GPUs[1][0])
                    self.conv3_d2 = unet_cmplx.conv(2 * s, s, dilation=2).cuda(GPUs[1][0])

                    self.conv4 = unet_cmplx.conv(2 * s, s).cuda(GPUs[2][0])
                    self.conv4_d3 = unet_cmplx.conv(2 * s, s, dilation=3).cuda(GPUs[2][0])

                    self.conv5 = unet_cmplx.conv(2 * s, s).cuda(GPUs[3][0])
                    self.conv5_d3 = unet_cmplx.conv(2 * s, s, dilation=3).cuda(GPUs[3][0])

                    self.conv6 = unet_cmplx.conv(2 * s, s).cuda(GPUs[0][0])
                    self.out_conv = unet_cmplx.outconv(s, 1).cuda(GPUs[0][0])

                    if DataParallel:
                        self.conv1 = torch.nn.DataParallel(self.conv1, device_ids=GPUs[0])
                        self.conv2 = torch.nn.DataParallel(self.conv2, device_ids=GPUs[0])
                        self.conv2_d2 = torch.nn.DataParallel(self.conv2_d2, device_ids=GPUs[0])
                        self.conv3 = torch.nn.DataParallel(self.conv3, device_ids=GPUs[1])
                        self.conv3_d2 = torch.nn.DataParallel(self.conv3_d2, device_ids=GPUs[1])
                        self.conv4 = torch.nn.DataParallel(self.conv4, device_ids=GPUs[2])
                        self.conv4_d3 = torch.nn.DataParallel(self.conv4_d3, device_ids=GPUs[2])
                        self.conv5 = torch.nn.DataParallel(self.conv5, device_ids=GPUs[3])
                        self.conv5_d3 = torch.nn.DataParallel(self.conv5_d3, device_ids=GPUs[3])
                        self.conv6 = torch.nn.DataParallel(self.conv6, device_ids=GPUs[0])
                        self.out_conv = torch.nn.DataParallel(self.out_conv, device_ids=GPUs[0])
        #             self.inc = unet_cmplx.inconv(n_channels, 32*s).to(torch.device('cuda:0'))
        #             self.down1 = unet_cmplx.down(32*s, 64*s).to(torch.device('cuda:1'))
        #             self.down2 = unet_cmplx.down(64*s, 128*s).to(torch.device('cuda:1'))
        #             self.bottleneck = unet_cmplx.bottleneck(128*s,128*s, False).to(torch.device('cuda:2'))
        #             self.up2 = unet_cmplx.up(128*s, 64*s).to(torch.device('cuda:2'))
        #             self.up3 = unet_cmplx.up(64*s, 32*s).to(torch.device('cuda:2'))
        #             self.up4 = unet_cmplx.up(32*s, 32*s).to(torch.device('cuda:3'))
        #             self.ouc = unet_cmplx.outconv(32*s, n_channels).to(torch.device('cuda:3'))
        elif params.MODEL == 3.4:  # stack of complex convolutional layers with RBN and CReLU (whole volume)
            s = 32
            self.num_GPUs = 3
            self.multiGPU = False
            DataParallel = True

            if self.num_GPUs == 3:
                GPUs = [[0, 5], [1, 4], [2, 3]]  # [[0,1,2],[5,4,3]] #[5,7,6]
                #                 self.conv_01 = unet_cmplx.conv(n_channels, s).cuda(GPUs[0][0])
                #                 self.conv_02 = unet_cmplx.conv(s, s).cuda(GPUs[0][0])
                #                 self.conv_03 = unet_cmplx.conv(s, s).cuda(GPUs[0][0])
                #                 self.conv_11 = unet_cmplx.conv(s, s).cuda(GPUs[1][0])
                #                 self.conv_12 = unet_cmplx.conv(s, s).cuda(GPUs[1][0])
                #                 self.conv_13 = unet_cmplx.conv(s, s).cuda(GPUs[1][0])
                #                 self.conv_21 = unet_cmplx.conv(s, s).cuda(GPUs[2][0])
                #                 self.conv_22 = unet_cmplx.conv(s, s).cuda(GPUs[2][0])
                #                 self.conv_23 = unet_cmplx.conv(s, s).cuda(GPUs[2][0])

                self.part1 = nn.Sequential(
                    unet_cmplx.conv(n_channels, s),
                    unet_cmplx.conv(s, s)
                ).cuda(GPUs[0][0])

                self.part2 = nn.Sequential(
                    unet_cmplx.conv(s, s),
                    unet_cmplx.conv(s, s)
                ).cuda(GPUs[1][0])

                self.part3 = nn.Sequential(
                    unet_cmplx.conv(s, s),
                    unet_cmplx.outconv(s, 1)
                ).cuda(GPUs[2][0])
                #
                #
                if DataParallel:
                    self.part1 = nn.DataParallel(self.part1, device_ids=GPUs[0])
                    self.part2 = nn.DataParallel(self.part2, device_ids=GPUs[1])
                    self.part3 = nn.DataParallel(self.part3, device_ids=GPUs[2])
        #                     self.conv_01 = unet_cmplx.conv(n_channels, s).cuda(GPUs[0])
        #                     self.conv_02 = unet_cmplx.conv(s, s).cuda(GPUs[0])
        #                     self.conv_03 = unet_cmplx.conv(s, s).cuda(GPUs[0])
        #                     self.conv_11 = unet_cmplx.conv(s, s).cuda(GPUs[1])
        #                     self.conv_12 = unet_cmplx.conv(s, s).cuda(GPUs[1])
        #                     self.conv_13 = unet_cmplx.conv(s, s).cuda(GPUs[1])
        #                     self.conv_21 = unet_cmplx.conv(s, s).cuda(GPUs[2])
        #                     self.conv_22 = unet_cmplx.conv(s, s).cuda(GPUs[2])
        #                     self.conv_23 = unet_cmplx.conv(s, s).cuda(GPUs[2])
        elif params.MODEL == 4:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            #             self.inc1 = inconv(64, 64)
            #             self.inc2 = inconv(64, 1)
            self.down1 = unet_cmplx.down(64, 128)
            self.down2 = unet_cmplx.down(128, 256)
            #             self.down3 = down(256, 512)
            self.bottleneck = unet_cmplx.bottleneck(256, 256)
            #             self.down4 = down(512, 512)
            #             self.up1 = up(1024, 256)
            self.up2 = unet_cmplx.up(256, 128)
            self.up3 = unet_cmplx.up(128, 64)
            self.up4 = unet_cmplx.up(64, n_channels)
        #             self.outc = outconv(n_channels, n_channels)
        #             self.mpcomb = mag_phase_combine(2, n_classes)
        elif params.MODEL == 5:
            self.inc = unet_cmplx.inconv(n_channels, 64)
            self.down1 = unet_cmplx.down(64, 128)
            self.down2 = unet_cmplx.down(128, 128)
            self.up3 = unet_cmplx.up(256, 128)
            self.up4 = unet_cmplx.up(192, 64)
            self.outc = unet_cmplx.outconv(64, n_channels)
            self.coilcmp = unet_cmplx.outconv(n_channels, 1)

        elif params.MODEL == 6:

            self.fc = torch.nn.Sequential(
                ComplexLinear(img_size[0] * img_size[1], img_size[0] * img_size[1])  # ,
                #           torch.nn.ReLU()#,
                # ComplexLinear(img_size[0]*img_size[1], img_size[0]*img_size[1])
            )

        elif params.MODEL == 7:  # Deep Unet
            ps = 2
            self.inc = unet_real.inconv(n_channels, 64 * ps)
            self.down1 = unet_real.down(64 * ps, 128 * ps)
            self.down2 = unet_real.down(128 * ps, 256 * ps)
            self.bottleneck = unet_real.bottleneck(256 * ps, 256 * ps, False)
            self.up2 = unet_real.up(256 * ps, 128 * ps)
            self.up3 = unet_real.up(128 * ps, 64 * ps)
            self.up4 = unet_real.up(64 * ps, 64)
            self.outc = unet_real.outconv(64, 1)

        elif params.MODEL == 8:  # Complex Network takes kspace input and image domain output
            self.stc_k1 = unet_cmplx.stacked_convs_block(n_channels, n_classes, kernel_size=11, dilations=(2, 3),
                                                         apply_BN=False, kspace=True)


        elif params.MODEL == 9:  # Complex Network takes neighborhood matrix input and image domain output
            # self.grid_kernel = GaussianGriddingKernels(kernel_mat_size=params.img_size)

            # self.stc_k1 = unet_cmplx.stacked_convs_block(n_channels, n_channels, kernel_size=5, dilations=1,
            #                                              apply_BN=False, apply_activation=False)
            # self.stc_k2 = unet_cmplx.stacked_convs_block(n_channels, n_channels, kernel_size=5, dilations=1,
            #                                              apply_BN=False, apply_activation=False)

            self.stc3D_k1 = unet_cmplx.stacked_3Dconvs_block(n_channels, n_channels, kernel_size=(3,5,5), dilations=1,
                                                          apply_BN=False, apply_activation=False)
            self.stc3D_k2 = unet_cmplx.stacked_3Dconvs_block(n_channels, n_channels, kernel_size=(3,5,5), dilations=1,
                                                          apply_BN=False, apply_activation=False)
            #
            # self.conv_3D_1 = unet_cmplx.conv_3D(n_channels, 32, kernel_size=(3, 3, 3), dilation=1, apply_BN=False,
            #                          apply_activation=True)
            # self.conv_3D_2 = unet_cmplx.conv_3D(32, 64,
            #                          kernel_size=(3, 3, 3), dilation=1, apply_BN=False, apply_activation=True)
            #
            # self.conv_3D_3 = unet_cmplx.conv_3D(64, 32,
            #                          kernel_size=(3, 3, 3), dilation=1, apply_BN=False, apply_activation=True)
            #
            # self.conv_3D_4 = unet_cmplx.conv_3D(32, n_classes,
            #                          kernel_size=(3, 3, 3), dilation=1, apply_BN=False, apply_activation=True)

            num_t1_w = 5;
            # # U-net
            self.unet_img = unet_cmplx.CUNet(n_channels, num_t1_w)

            self.t1_fit_1 = unet_cmplx.conv( 2*num_t1_w, num_t1_w, kernel_size=3, dilation=1, bias=True, groups=1, apply_BN=True, apply_activation=False)
            self.t1_fit_2 = unet_cmplx.conv( num_t1_w, num_t1_w//2, kernel_size=3, dilation=1, bias=True, groups=1, apply_BN=True, apply_activation=False)
            self.t1_out = unet_cmplx.outconv(num_t1_w//2, 1)
        elif params.MODEL == 10:
            self.T1fitNet = nn.Sequential(
            nn.Linear(in_features=n_channels, out_features=400),
            nn.LeakyReLU(),
            nn.Linear(in_features=400, out_features=400),
            nn.LeakyReLU(),
            nn.Linear(in_features=400, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=1),

            )

            # self.T1fitNet = nn.Sequential(
            #     nn.Linear(in_features=n_channels, out_features=1000),
            #     nn.LeakyReLU(),
            #     nn.Linear(in_features=1000, out_features=1000),
            #     nn.LeakyReLU(),
            #     nn.Linear(in_features=1000, out_features=1)
            # )
            self.one_fcl = nn.Linear(100, 1)
            in_ch = 100
            inter_ch = 64

            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_ch, inter_ch, 3, padding=1),
                # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_ch, inter_ch, 5, padding=2),
                # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_ch, inter_ch, 5, padding=2),
                # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_ch, 1, 3, padding=1)
                # nn.BatchNorm2d(out_ch, affine=True), #, affine=False
            )

    def forward(self, x, apply_conv=False):

        if params.MODEL == 0:  # Deep Unet
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            x = self.mpcomb(x)

        elif params.MODEL == 1:  # Shallow Unet
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            #         x4 = self.down3(x3)
            #         x5 = self.down4(x4)
            #         x = self.up1(x5, x4)
            #         x = self.up2(x, x3)
            x = self.up3(x3, x2)
            #         saveArrayToMat(x.cpu().data.numpy()[0,:,:,:], 'x', 'bx/{:.0f}'.format(time.time()))
            x = self.up4(x, x1)
            #         saveArrayToMat(x.cpu().data.numpy()[0,:,:,:], 'x', 'x/{:.0f}'.format(time.time()))
            x = self.outc(x)  # residual
            if not params.magnitude_only:
                x = self.mpcomb(x)

        elif params.MODEL == 2:  # COMPLEX CONV NET - Only Stacked Convolution layers
            x0 = x
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
            x = self.outc(x) + x0  #### Residual connection

        elif params.MODEL == 3:
            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)  # + x0
            x = self.ouc(x)

        elif params.MODEL == 3.1:
            x = self.inc(x)
            x = self.conv64(x)
            #             x = self.conv64(x)
            x = self.conv64_32(x)
            x = self.conv32(x)
            #             x = self.conv32(x)
            x = self.conv32_64(x)
            x = self.conv64(x)
            #             x = self.conv64(x)
            x = self.outc(x)

        elif params.MODEL == 3.2:
            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)  # + x0
        #             x = self.ouc(x)
        elif params.MODEL == 3.3:

            if not self.multiGPU:
                x = self.conv1(x)
                x = torch.cat([self.conv2(x), self.conv2_d2(x)], 1)
                x = torch.cat([self.conv3(x), self.conv3_d2(x)], 1)
                x = torch.cat([self.conv4(x), self.conv4_d3(x)], 1)
                x = torch.cat([self.conv5(x), self.conv5_d3(x)], 1)
                x = self.conv6(x)
                x = self.out_conv(x)

            else:
                if self.num_GPUs == 3:
                    ## 1st GPU
                    x = self.conv1(x)
                    x = torch.cat([self.conv2(x), self.conv2_d2(x)], 1)
                    x3 = self.conv3(x)

                    ## 2nd GPU
                    nxt_gpu = x.device.index + 1;
                    x = x.cuda(nxt_gpu)
                    x = torch.cat([x3.cuda(nxt_gpu), self.conv3_d2(x)], 1)
                    x = torch.cat([self.conv4(x), self.conv4_d3(x)], 1)

                    ## 3rd GPU
                    nxt_gpu = x.device.index + 1;
                    x = x.cuda(nxt_gpu)
                    x = torch.cat([self.conv5(x), self.conv5_d3(x)], 1)
                    x = self.conv6(x)

                    ## 1st GPU
                    nxt_gpu = x.device.index - 2;
                    x = x.cuda(nxt_gpu)
                    x = self.out_conv(x)

                elif self.num_GPUs == 4:
                    ## 1st GPU
                    x = self.conv1(x)
                    x = torch.cat([self.conv2(x), self.conv2_d2(x)], 1)

                    ## 2nd GPU
                    nxt_gpu = x.device.index + 1;
                    x = x.cuda(nxt_gpu)
                    x = torch.cat([self.conv3(x), self.conv3_d2(x)], 1)

                    ## 3rd GPU
                    nxt_gpu = x.device.index + 1;
                    x = x.cuda(nxt_gpu)
                    x = torch.cat([self.conv4(x), self.conv4_d3(x)], 1)

                    ## 4th GPU
                    nxt_gpu = x.device.index + 1;
                    x = x.cuda(nxt_gpu)
                    x = torch.cat([self.conv5(x), self.conv5_d3(x)], 1)

                    ## 1st GPU
                    nxt_gpu = x.device.index - 3;
                    x = x.cuda(nxt_gpu)
                    x = self.conv6(x)
                    x = self.out_conv(x)

        elif params.MODEL == 3.4:
            nxt_gpu = x.device.index + 1
            #             print(x.device.index)
            x = self.part1(x).cuda(nxt_gpu)
            nxt_gpu = x.device.index + 1
            #             print(x.device.index)
            x = self.part2(x).cuda(nxt_gpu)
            #             print(x.device.index)
            x = self.part3(x).cuda(nxt_gpu - 2)

        elif params.MODEL == 4:

            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)  # the residual connection is added inside
            x = self.up2(x4, x3) + down_x2
            x = self.up3(x, x2) + down_x1
            x = self.up4(x, x1) + x0
        #             x = self.outc(x) + x0 #### Residual connection

        elif params.MODEL == 5:
            x0 = x
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
            x = self.outc(x) + x0  #### Residual connection
            x = self.coilcmp(x)

        elif params.MODEL == 6:
            x = self.fc(x)

        elif params.MODEL == 7:
            x0 = x
            x1 = self.inc(x)
            x2, down_x1 = self.down1(x1)
            x3, down_x2 = self.down2(x2)
            x4 = self.bottleneck(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)  # + x0
            x = self.outc(x)

        elif params.MODEL == 8:  # Complex Network takes kspace input and image domain output
            x = self.stc_k1(x)

        elif params.MODEL == 9:  # Complex Network takes neighborhood matrix input and image domain output
            # saveTensorToMat(fftshift2d(torch.ifft(x, 2, normalized=True), [3, 4]), 'x')
            # shape = x.shape
            # x = x.reshape((shape[0], shape[1] * shape[2], shape[3], shape[4], shape[5]))

            # saveTensorToMat(fftshift2d(torch.ifft(x, 2, normalized=True), [3, 4]), 'x0')
            # saveTensorToMat(x, 'k0')

            x = self.stc3D_k1(x) + x
            # saveTensorToMat(fftshift2d(torch.ifft(x, 2, normalized=True), [3, 4]), 'x1')
            # saveTensorToMat(x, 'k1')

            x = self.stc3D_k2(x) + x
            # saveTensorToMat(fftshift2d(torch.ifft(x, 2, normalized=True), [3, 4]), 'x2')
            # saveTensorToMat(x, 'k2')

            #
            x = torch.ifft(x, 2, normalized=True)
            x = normalizeComplexBatch_byMagnitudeOnly(x, normalize_over_channel=True)

            # x = self.conv_3D_1(x)
            # x = self.conv_3D_2(x)
            # x = self.conv_3D_3(x)
            # x = self.conv_3D_4(x)

            # saveTensorToMat(x, 'x1')

            # x = x.reshape((shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]))

            # saveTensorToMat(x, 'x2')
            # x = self.stc_Im1_1(x) + x
            # x = self.stc_Im1_2(x) + x
            #
            # x = self.stc_Im2_1(x) + x
            # x = self.stc_Im2_2(x) + x
            #
            # x = self.stc_Im3_1(x) + x
            # x = self.stc_Im3_2(x) + x
            x = self.unet_img(x)
            x = torch.cat((x, TI), 1)
            # saveTensorToMat(x, 't1w')
            
            x = self.t1_fit_1(x)
            x = self.t1_fit_2(x)
            x = self.t1_out(x)
        elif params.MODEL == 10:
            x = self.T1fitNet(x)
            # if apply_conv:
            #     xs = int(x.shape[0]/(128*128))
            #     x = x.reshape(xs, 128,128,1,100)
            #     x = x.permute((0,3,4,1,2)).squeeze(1)
            #     x = self.conv_layers(x)
            # else:
            #     x = self.one_fcl(x)
        return x


