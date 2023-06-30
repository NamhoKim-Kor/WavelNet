import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


class WaveletConv(nn.Module):
    """
    Wavelet-based convolution

    Parameters
    ----------
    n_channels_in : 'int'
        Number of input channels. Must be 1.
    n_channels_out : 'int'
        Number of filters.
    size_kernel : 'int'
        Filter length. Must be an odd number.
    f_sampling : 'int'
        Sampling frequency of input signal.
    f_cufoff_min: 'int'
        Minimal cut-off frequency of filter.
    f_cufoff_min: 'int'
        Maximal cut-off frequency of filter.

    Usage
    -----
    See `torch.nn.Conv1d`
    m = SincConv_NHK(n_channels_in=720,
                     n_channels_out=64,
                     size_kernel=360,
                     f_sampling=360,
                     f_cutoff_min=1.5,
                     f_cutoff_max=64)
    input = torch.randn(50,1,720)
    features = m(input)

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    def __init__(self, in_channels, out_channels, kernel_size, fs, mother_wavelet, fs_wavelet, fc_wavelet, a_min=None,
                 stride=1, dilation=1, bias=None, groups=1):
        super(WaveletConv, self).__init__()

        if in_channels != 1:
            msg = 'WaveletConv only supports one input channel (here, n_channels_input = {%i})'\
                  % (in_channels)
            raise ValueError(msg)
        if kernel_size % 2 != 1:
            msg = 'WaveletConv only supports an odd number as the size of kernel (here, size_kernel = {%i})'\
                  % (kernel_size)
            raise ValueError(msg)
        if bias:
            raise ValueError('WaveletConv does not support bias.')
        if groups > 1:
            raise ValueError('WaveletConv does not support groups.')

        if np.abs(np.sum(mother_wavelet)) > 0.01:
            print(np.abs(np.sum(mother_wavelet)))
            raise ValueError('Mother wavelet does not satisfy zero mean condition.')
        if np.abs(np.sum(mother_wavelet**2) - 1) > 0.01:
            print(np.abs(np.sum(mother_wavelet**2) - 1))
            raise ValueError('Mother wavelet does not satisfy square norm one condition.')
        if not a_min is None:
            if a_min >= 1 or a_min <= 0:
                raise ValueError('Minimum scale parameter should be larger than 0 and smaller 1.')
        if a_min is None:
            # determine the lower bound of scale parameter
            a_min = max(fc_wavelet / (fs/2), 11/kernel_size)

        self.n_channels_in = in_channels
        self.n_channels_out = out_channels
        self.size_kernel = kernel_size
        self.mother_wavelet = mother_wavelet
        self.a_min = a_min
        self.stride = stride
        self.padding = int(np.floor(self.size_kernel/2))
        self.dilation = dilation
        self.bias = bias
        self.groups = groups

        # initialize scale parameters of wavelets (which would be used as kernels)
        # they are evenly spaced
        a_init = np.linspace(self.a_min, 1, self.n_channels_out)

        # set scale parameters to be learnable
        # a : (n_channels_out, 1)
        self.a = torch.nn.Parameter(torch.Tensor(a_init).view(-1, 1))

        self.mother_wavelet = torch.tensor(self.mother_wavelet)

    def forward(self, input):
        """
        Wavelet-based convolution
        Optimized by Namho Kim

        Parameters
        ----------
        input : 'torch.Tensor' (batch_size, 1, n_samples)
            Batch of input signals.

        Returns
        -----
        features : 'torch.Tensor' (batch_size, n_channels_out, n_samples)
            Batch of wavelet filters activations.
        """
        self.mother_wavelet = self.mother_wavelet.to(input.device)

        # Generate wavelets as kernels
        a_prac = torch.clamp(input=self.a, min=self.a_min, max=1)
        a_prac = a_prac.to(input.device)
        wavelet_kernels = scale_wavelet(self.mother_wavelet, a_prac, self.size_kernel)

        # plt.figure(figsize=(20,18))
        # for i in range(9):
        #     plt.subplot(3,3,i+1)
        #     plt.title(str(a_prac[-i]))
        #     plt.plot(np.linspace(0, len(wavelet_kernels.detach().cpu()[-i]), len(wavelet_kernels.detach().cpu()[-i])), wavelet_kernels.detach().cpu()[-i])
        # plt.tight_layout()
        # plt.show()
        # print(a_prac)

        self.wavelet_kernels_final = (wavelet_kernels).view(self.n_channels_out, 1, self.size_kernel)

        return nn.functional.conv1d(input=input,
                                    weight=self.wavelet_kernels_final,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    bias=self.bias,
                                    groups=self.groups)


def scale_wavelet(mother_wavelet, a, len_wavelet):
    wavelet_scaled = torch.zeros(a.size(dim=0), len_wavelet)
    wavelet_scaled = wavelet_scaled.to(mother_wavelet.device)
    mother_wavelet = (mother_wavelet).view(1, 1, len_wavelet)
    for i in range(a.size(dim=0)):
        wavelet_scaled_i = F.interpolate(mother_wavelet, scale_factor=a[i].item(), recompute_scale_factor=True)
        wavelet_scaled_i = wavelet_scaled_i.reshape(-1)
        len_scaled = len(wavelet_scaled_i)
        offset = (len_wavelet - len_scaled) // 2
        wavelet_scaled[i][offset:offset+len_scaled] = torch.div(wavelet_scaled_i, torch.sqrt(a[i]))

    return wavelet_scaled