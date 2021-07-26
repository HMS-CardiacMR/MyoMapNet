
import torch
import numpy as np
import torch.nn as nn
from saveNet import *
from math import exp

def sinc(input, weights):
    x, y = torch.unbind(input, dim=-1)
    b, c = torch.unbind(weights, dim=-1)
    mag = np.pi * ((b*x)**2 + (c*y)**2)**0.5
    output = torch.sin(mag)/mag
    return output

class GriddingKernels(nn.Module):
    def __init__(self, kernel_mat_size=(416, 416), eps=0.0001, init_densiy=None, init_kernel_param=None):
        super(GriddingKernels, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones((kernel_mat_size[0], kernel_mat_size[1])), requires_grad=True)
        self.b = nn.Parameter(torch.ones((kernel_mat_size[0], kernel_mat_size[1], 2)), requires_grad=True) #for x and y

        self.reset_density_comp_params(init_densiy)

        self.reset_kernel_params(init_kernel_param)

    def reset_density_comp_params(self, init_densiy=None, s=0.025, bias=0.125):
        if init_densiy is None:
            for i in range(0, self.a.shape[0]):
                for j in range(0, self.a.shape[1]):
                    self.a.data[i, j] = s * (
                                (i - self.a.shape[0] // 2) ** 2 + (j - self.a.shape[1] // 2) ** 2) ** 0.5 + bias
        else:
            self.a.data = init_densiy

    def reset_kernel_params(self, init_kernel_param=None, s=0.005, bias=0.05):
        if init_kernel_param is None:

            for i in range(0, self.b.shape[0]):
                for j in range(0, self.b.shape[1]):
                    self.b.data[i, j, 0] = s * (
                                (i - self.b.shape[0] // 2) ** 2 + (j - self.b.shape[1] // 2) ** 2) ** 0.5 + bias
                    self.b.data[i, j, 1] = s * (
                                (i - self.b.shape[0] // 2) ** 2 + (j - self.b.shape[1] // 2) ** 2) ** 0.5 + bias
                    # self.b.data[i, j, 2] = s * ((i-self.a.shape[0]//2) ** 2 + (j-self.a.shape[1]//2) ** 2) ** 0.5 + 2 * s
        else:
            self.a.data = init_kernel_param


    # def reset_density_comp_params(self, init_densiy=None):
    #     if init_densiy is None:
    #         r = 10/(self.a.shape[0]//2)
    #         for i in range(0, self.a.shape[0]):
    #             for j in range(0, self.a.shape[1]):
    #                 self.a.data[i, j] = r * ((i-self.a.shape[0]//2)**2 + (j-self.a.shape[1]//2)**2)**0.5 + 2*r
    #     else:
    #         self.a.data = init_densiy
    #
    # def reset_kernel_params(self, init_kernel_param=None):
    #     if init_kernel_param is None:
    #         kernel_window_size = 10
    #         s = (self.b.shape[0]//2) / kernel_window_size
    #
    #         for i in range(0, self.b.shape[0]):
    #             for j in range(0, self.b.shape[1]):
    #                 self.b.data[i, j, 0] = s / ((i-self.a.shape[0]//2) ** 2 + (j-self.a.shape[1]//2) ** 2 + 1) ** 0.5
    #                 self.b.data[i, j, 1] = s / ((i-self.a.shape[0]//2) ** 2 + (j-self.a.shape[1]//2) ** 2 + 1) ** 0.5
    #                 # self.b.data[i, j, 2] = s * ((i-self.a.shape[0]//2) ** 2 + (j-self.a.shape[1]//2) ** 2) ** 0.5 + 2 * s
    #     else:
    #         self.a.data = init_kernel_param

    def forward(self, Ksp_ri, Loc_xy):
        Loc_xy = Loc_xy + self.eps
        dims = Ksp_ri.shape
        brdcst_a = self.a.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand((dims[0], dims[1], dims[3], dims[4], dims[5]))
        brdcst_b = self.b.unsqueeze(0).unsqueeze(0).expand_as(Loc_xy)

        kernel = sinc(Loc_xy, brdcst_b).unsqueeze(1).unsqueeze(-1).expand_as(Ksp_ri)

        output = brdcst_a * torch.mean(Ksp_ri * kernel, dim=2)
        
        return output




def gaussian(input, weights):
    x, y = torch.unbind(input, dim=-1)
    b, c = torch.unbind(weights, dim=-1)
    output = torch.exp(-( (x/b)**2 + (y/c)**2))
    return output

class GaussianGriddingKernels(nn.Module):
    def __init__(self, kernel_mat_size=(416, 416), init_densiy=None, init_kernel_param=None):
        super(GaussianGriddingKernels, self).__init__()
        self.a = nn.Parameter(torch.ones((kernel_mat_size[0], kernel_mat_size[1])), requires_grad=False)
        self.b = nn.Parameter(torch.ones((kernel_mat_size[0], kernel_mat_size[1], 2)),
                              requires_grad=True)  # for x and y

        self.reset_density_comp_params(init_densiy)
        self.reset_kernel_params(init_kernel_param)

    def reset_density_comp_params(self, init_densiy=None, s=1/208, bias=0.02):
        if init_densiy is None:
            for i in range(0, self.a.shape[0]):
                for j in range(0, self.a.shape[1]):
                    self.a.data[i, j] = s * (
                                (i - self.a.shape[0] // 2) ** 2 + (j - self.a.shape[1] // 2) ** 2) ** 0.5 + bias
        else:
            self.a.data = init_densiy

    def reset_kernel_params(self, init_kernel_param=None, s=0.0002, bias=1):
        if init_kernel_param is None:

            for i in range(0, self.b.shape[0]):
                for j in range(0, self.b.shape[1]):
                    self.b.data[i, j, 0] = s * (
                                (i - self.b.shape[0] // 2) ** 2 + (j - self.b.shape[1] // 2) ** 2) ** 0.5 + bias
                    self.b.data[i, j, 1] = s * (
                                (i - self.b.shape[0] // 2) ** 2 + (j - self.b.shape[1] // 2) ** 2) ** 0.5 + bias
                    # self.b.data[i, j, 2] = s * ((i-self.a.shape[0]//2) ** 2 + (j-self.a.shape[1]//2) ** 2) ** 0.5 + 2 * s
        else:
            self.a.data = init_kernel_param

    def forward(self, Ksp_ri, Loc_xy):
        dims = Ksp_ri.shape
        # print('==================>',self.b[207,207,0].cpu().data.numpy())
        # brdcst_a = self.a.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand((dims[0], dims[1], dims[3], dims[4], dims[5]))
        brdcst_b = self.b.unsqueeze(0).unsqueeze(0).expand_as(Loc_xy)

        # x, y = torch.unbind(Loc_xy, dim=-1)
        # denst = (2*((x**2 + y**2)**0.5)/dims[-2]).unsqueeze(1).unsqueeze(-1).expand_as(Ksp_ri)
        # Ksp_ri = denst * Ksp_ri

        kernel = gaussian(Loc_xy, brdcst_b).unsqueeze(1).unsqueeze(-1).expand_as(Ksp_ri)

        # output = brdcst_a * torch.mean(Ksp_ri * kernel, dim=2)
        output = torch.mean(Ksp_ri * kernel, dim=2)

        return output
