# https://github.com/jbhuang0604/SelfExSR

# import torch
# import pandas as pd
from torch import nn  # , optim


# from torch.nn import functional as F
# from torch.utils.data.dataset import Dataset
# from torchvision import transforms
# from PIL import Image
#
# from sklearn.metrics import roc_curve, auc
# from matplotlib import pyplot as plt


class BasicGenBlock(nn.Module):

    def __init__(self, kernel_size=3, stride=1, channels=64, bias=True):
        super(BasicGenBlock, self).__init__()

        self.k_size = kernel_size
        self.padding = self.k_size // 2
        self.stride = stride
        self.bias = bias

        self.channels = channels

        self.layers = nn.Sequential(nn.Conv2d(in_channels=self.channels,
                                              out_channels=self.channels,
                                              kernel_size=self.k_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              bias=self.bias),
                                    nn.BatchNorm2d(self.channels),
                                    nn.PReLU(),
                                    nn.Conv2d(in_channels=self.channels,
                                              out_channels=self.channels,
                                              kernel_size=self.k_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              bias=self.bias),
                                    nn.BatchNorm2d(self.channels)
                                    )

    def forward(self, x):
        out = self.layers(x)

        return x + out


class Generator(nn.Module):

    def __init__(self, init_kernel_size=9, kernel_size=3, stride=1, channels=64, upscale_factor=2, bias=True):
        """
        Model initializer method.

        :param bias: Bias in system (default False).
        :param kernel_size: Convolution kernel size.
        """

        super(Generator, self).__init__()

        self.init_k_size = init_kernel_size
        self.init_padding = self.init_k_size // 2

        self.k_size = kernel_size
        self.padding = self.k_size // 2

        self.st = stride
        self.bias = bias

        self.intrim_channels = channels
        self.final_channels = channels * (upscale_factor ** 2)

        self.upscale_factor = upscale_factor

        # self.upsample_mode = 'nearest'
        # r"""
        # Upsampling algorithm: one of ``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'``x and ``'trilinear'``.
        # """

        self.init_layer = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=self.intrim_channels,
                                                  kernel_size=self.init_k_size,
                                                  stride=self.st,
                                                  padding=self.init_padding,
                                                  bias=self.bias),
                                        nn.PReLU()
                                        )

        self.blocks_layer = nn.Sequential(BasicGenBlock(kernel_size=self.k_size,
                                                        stride=self.st,
                                                        channels=self.intrim_channels,
                                                        bias=self.bias),
                                          BasicGenBlock(kernel_size=self.k_size,
                                                        stride=self.st,
                                                        channels=self.intrim_channels,
                                                        bias=self.bias),
                                          BasicGenBlock(kernel_size=self.k_size,
                                                        stride=self.st,
                                                        channels=self.intrim_channels,
                                                        bias=self.bias),
                                          BasicGenBlock(kernel_size=self.k_size,
                                                        stride=self.st,
                                                        channels=self.intrim_channels,
                                                        bias=self.bias),
                                          BasicGenBlock(kernel_size=self.k_size,
                                                        stride=self.st,
                                                        channels=self.intrim_channels,
                                                        bias=self.bias)
                                          )

        self.intrim_layer = nn.Sequential(nn.Conv2d(in_channels=self.intrim_channels,
                                                    out_channels=self.intrim_channels,
                                                    kernel_size=self.k_size,
                                                    stride=self.st,
                                                    padding=self.padding,
                                                    bias=self.bias),
                                          nn.BatchNorm2d(self.intrim_channels)
                                          )

        self.pixel_layer = nn.Sequential(nn.Conv2d(in_channels=self.intrim_channels,
                                                   out_channels=self.final_channels,
                                                   kernel_size=self.k_size,
                                                   stride=self.st,
                                                   padding=self.padding,
                                                   bias=self.bias),
                                         nn.PixelShuffle(upscale_factor=self.upscale_factor),
                                         nn.PReLU(),
                                         nn.Conv2d(in_channels=self.intrim_channels,
                                                   out_channels=self.final_channels,
                                                   kernel_size=self.k_size,
                                                   stride=self.st,
                                                   padding=self.padding,
                                                   bias=self.bias),
                                         nn.PixelShuffle(upscale_factor=self.upscale_factor),
                                         nn.PReLU()
                                         )
        """
        The input channels for both convolutions is 64 and output is 64 * (Scale_Factor ^ 2).
        """

        self.final_conv = nn.Conv2d(in_channels=self.intrim_channels,
                                    out_channels=3,
                                    kernel_size=self.init_k_size,
                                    stride=self.st,
                                    padding=self.init_padding,
                                    bias=self.bias)

    def forward(self, x):

        skip_var = self.init_layer(x)
        out = self.blocks_layer(skip_var)
        out = self.intrim_layer(out)
        out = self.pixel_layer(out + skip_var)
        out = self.final_conv(out)

        return out


class BasicDisBlock(nn.Module):

    def __init__(self, kernel_size=3, stride=1, in_channels=64, out_channels=64, bias=True):
        super(BasicDisBlock, self).__init__()

        self.k_size = kernel_size
        self.padding = self.k_size // 2
        self.stride = stride
        self.bias = bias

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.k_size,
                                             stride=self.stride,
                                             padding=self.padding,
                                             bias=self.bias),
                                   nn.BatchNorm2d(self.in_channels),
                                   nn.LeakyReLU(),
                                   )

    def forward(self, x):
        return self.layer(x)
