"""
This module contains the Stegnet model code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

import torch.nn
from collections import OrderedDict
def dw_conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=out_channels,
                      bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), get_norm(_NORM, out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]

def conv3x3(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]





def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)







class ResidualDenseBlock_out_drop(nn.Module):
    def __init__(self, input=3, output=3, bias=True, dropout_rate=0.2):
        super(ResidualDenseBlock_out_drop, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x1 = self.dropout(x1)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x2 = self.dropout(x2)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x3 = self.dropout(x3)
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = self.dropout(x4)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


from aEMA_attention_module import ScConv


class ResidualBlockNoBN2(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(3, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.scconv1 = ScConv(mid_channels)
        self.scconv2 = ScConv(mid_channels)

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

        self.conv11 = nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True)
    def init_weights(self):
       # N.init_weights(self.conv1, init_type='kaiming')
        #N.init_weights(self.conv2, init_type='kaiming')
        init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        self.conv1.weight.data *= 0.1
        self.conv2.weight.data *= 0.1

    def forward(self, x):
        identity = x
        out = self.scconv2(self.conv2(self.scconv1(self.relu(self.conv1(x)))))
        out = self.conv11(out)
        return identity + out * self.res_scale

#from cbam import CBAM
class SeparableConv2d(nn.Module):
	"""
	Source: https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
	"""

	def __init__(self, in_channels, out_channels, kernel_size, padding):

		super().__init__()

		self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
									padding=padding, groups=in_channels, bias=False)

		self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

	def forward(self, x):

		x = self.depthwise(x)
		x = self.pointwise(x)
		return x

class Spatial2Channel(nn.Module):
	"""
	Spatial2Channel的类，它是一个继承自nn.Module的神经网络模块。这个类的作用是在空间维度上将输入的通道数转换为输出的通道数。
	Derived based on:
	https://github.com/adamcavendish/Deep-Image-Steganography/blob/master/model/steg_net/steganography.py#L78
	"""

	def __init__(self, activation, in_channels, out_channels, kernel_size):

		super().__init__()

		assert in_channels <= out_channels

		self.padding = (0, 0, 0, 0, 0, out_channels - in_channels)

		self.bnorm = nn.BatchNorm2d(in_channels)
		self.activation = activation

		self.sep_conv = SeparableConv2d(in_channels, out_channels, kernel_size, padding='same')

	def forward(self, x):

		padded_x = F.pad(x, self.padding, 'constant', 0)

		x = self.bnorm(x)
		x = self.activation(x)
		x = self.sep_conv(x)

		x = x + padded_x

		return x




class Stegnet(nn.Module):

	def __init__(self, in_channels):

		super().__init__()

		self.layer0 = SeparableConv2d(in_channels, 32, 3, padding='same')

		self.layer1 = Spatial2Channel(nn.ELU(), 32, 32, 3)
		self.layer2 = Spatial2Channel(nn.ELU(), 32, 64, 3)
		self.layer3 = Spatial2Channel(nn.ELU(), 64, 64, 3)
		self.layer4 = Spatial2Channel(nn.ELU(), 64, 128, 3)
		self.layer5 = Spatial2Channel(nn.ELU(), 128, 128, 3)
		self.bnorm5 = nn.BatchNorm2d(128)
		self.activ5 = nn.ELU()



		self.layer6 = nn.Conv2d(128, 32, 3, padding='same')
		self.bnorm6 = nn.BatchNorm2d(32)
		self.activ6 = nn.ELU()
		self.layer7 = nn.Conv2d(32, 3, 3, padding='same')

	def forward(self, x):


		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.bnorm5(x)
		x = self.activ5(x)



		x = self.layer6(x)
		x = self.bnorm6(x)
		x = self.activ6(x)
		x = self.layer7(x)

		return x