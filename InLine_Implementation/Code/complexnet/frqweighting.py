import torch
from torch.nn.parameter import Parameter

import numpy as np
import torch.nn as nn
from utils.polarTransforms import *
from saveNet import *


class FrequencyWeighting(nn.Module):

    def __init__(self, rank, grid, center=(0, 0), sigma=(1, 1), polar=False):
        super(FrequencyWeighting, self).__init__()
        self.rank = rank
        self.grid = grid #grid is
        self.center = center
        self.sigma = sigma
        self.polar = polar
        self.amplitude = 1

        self.gauss_2D = np.exp(-((self.grid[0] - self.center[0]) / 2 * self.sigma[0]**2 +
                                    (self.grid[1] - self.center[1]) / 2 * self.sigma[1]**2)
                                  )

    def forward(self, input, inverse=True):
        ndims = input.ndimension()
        input_real = input.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        input_imag = input.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)

        if not self.polar:
            mag, phase = cylindricalToPolarConversion(input_real, input_imag)
        else:
            mag = input_real
            phase = input_imag

        self.gauss_2D = self.gauss_2D.expand_as(mag)
        dims = mag.shape

        self.amplitude = torch.max( torch.reshape(mag, (dims[0], dims[1], dims[2] * dims[3])), axis=2, keep_dim=True)

        if inverse:
            output_mag_weighted = 1 / (self.amplitude * self.gauss_2D)
        else:
            output_mag_weighted = self.amplitude * self.gauss_2D

        if not self.polar:
            output_real, output_imag = polarToCylindricalConversion(output_mag_weighted, phase)
        else:
            output_real = output_mag_weighted
            output_imag = phase

        output = torch.stack((output_real, output_imag), dim=ndims - 1)

        return output


class FrequencyWeighting1d(FrequencyWeighting):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        n_channel: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self,
                 grid,
                 center=(0, 0),
                 sigma=(1, 1),
                 polar=False):
        # super(FrequencyWeighting1d, self).__init__()
        raise NotImplementedError


class FrequencyWeighting2d(FrequencyWeighting):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        n_channel: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self,
                 grid,
                 center=(0, 0),
                 sigma=(1, 1),
                 polar=False):
        super(FrequencyWeighting2d, self).__init__(
            rank=2,
            grid=grid,
            center=center,
            sigma=sigma,
            polar=polar
        )


class FrequencyWeighting3d(FrequencyWeighting):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        n_channel: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self,
                 grid,
                 center=(0, 0),
                 sigma=(1, 1),
                 polar=False):
        # super(FrequencyWeighting3d, self).__init__()
        raise NotImplementedError




