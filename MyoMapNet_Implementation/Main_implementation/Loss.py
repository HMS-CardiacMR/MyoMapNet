
from math import exp

import torch
from torch.autograd import Function, Variable
from torch.nn.modules.loss import _Loss

import numpy as np
import torch.nn.functional as F
from saveNet import *


class weighted_mse(_Loss):
    def __init__(self):
        super(weighted_mse, self).__init__()

    def forward(self, input, output, weight):
        return torch.sum(weight * (input - output) ** 2) / input.numel()

class weighted_mae(_Loss):
    def __init__(self):
        super(weighted_mae, self).__init__()
    def forward(self, input, output, weight):
        tmp = weight[weight>0]
        return torch.sum(weight * torch.abs(input - output)) / tmp.numel()


