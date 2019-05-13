# https://github.com/jbhuang0604/SelfExSR
# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

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

    def __init__(self, kernel_size=3, stride=1, channels=64, squeeze_factor=2, bias=True):
        super(BasicGenBlock, self).__init__()

        self.k_size = kernel_size
        self.padding = self.k_size // 2
        self.stride = stride
        self.bias = bias

        self.squeeze_factor = squeeze_factor

        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

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

        self.squeeze_net = nn.Sequential(nn.Linear(in_features=self.channels,
                                                   out_features=self.channels // self.squeeze_factor,
                                                   bias=self.bias),
                                         nn.Linear(in_features=self.channels // self.squeeze_factor,
                                                   out_features=self.channels,
                                                   bias=self.bias)
                                         )

    def forward(self, x):

        out = self.layers(x)

        batch_size, channels, _, _ = x.size()
        sq = self.avg_pool(x).view(batch_size, channels)
        sq = self.squeeze_net(sq)

        # sq = self.squeeze_net()

        return out * sq.view(batch_size, channels, 1, 1).expand_as(out)


class Generator2(nn.Module):

    def __init__(self, init_kernel_size=9, kernel_size=3, stride=1, channels=64, upscale_factor=2, bias=True):
        """
        Model initializer method.

        :param bias: Bias in system (default False).
        :param kernel_size: Convolution kernel size.
        """

        super(Generator2, self).__init__()

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


class Discriminator(nn.Module):

    def __init__(self, image_size=(224, 224), bias=False):
        super(Discriminator, self).__init__()

        self.k_size = 3
        self.padding = self.k_size // 2
        self.bias = bias

        # else:
        self.input_image_size = image_size
        if len(image_size) != 2:
            raise ValueError("Input Image size must be a tuple (Width x Height)")

        self.flattened_feat = (self.input_image_size[0] // 16) * (self.input_image_size[1] // 16)

        self.init_layer = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=64,
                                                  kernel_size=self.k_size,
                                                  stride=1,
                                                  padding=self.padding,
                                                  bias=self.bias),
                                        nn.LeakyReLU()
                                        )

        self.blocks_layer = nn.Sequential(BasicDisBlock(kernel_size=self.k_size,
                                                        stride=2,
                                                        in_channels=64,
                                                        out_channels=64,
                                                        bias=self.bias),
                                          BasicDisBlock(kernel_size=self.k_size,
                                                        stride=1,
                                                        in_channels=64,
                                                        out_channels=128,
                                                        bias=self.bias),
                                          BasicDisBlock(kernel_size=self.k_size,
                                                        stride=2,
                                                        in_channels=128,
                                                        out_channels=128,
                                                        bias=self.bias),
                                          BasicDisBlock(kernel_size=self.k_size,
                                                        stride=1,
                                                        in_channels=128,
                                                        out_channels=256,
                                                        bias=self.bias),
                                          BasicDisBlock(kernel_size=self.k_size,
                                                        stride=2,
                                                        in_channels=256,
                                                        out_channels=256,
                                                        bias=self.bias),
                                          BasicDisBlock(kernel_size=self.k_size,
                                                        stride=1,
                                                        in_channels=256,
                                                        out_channels=512,
                                                        bias=self.bias),
                                          BasicDisBlock(kernel_size=self.k_size,
                                                        stride=2,
                                                        in_channels=512,
                                                        out_channels=512,
                                                        bias=self.bias),
                                          )

        self.linear_layer = nn.Sequential(nn.Linear(in_features=self.flattened_feat,
                                                    out_features=1024,
                                                    bias=self.bias),
                                          nn.LeakyReLU()
                                          )

        self.classifier = nn.Sequential(nn.Linear(in_features=1024,
                                                  out_features=1,
                                                  bias=self.bias),
                                        nn.Sigmoid()
                                        )

    def forward(self, x):

        out = self.init_layer(x)
        out = self.blocks_layer(out)
        out = self.linear_layer(out)
        out = self.classifier(out)

        return out
