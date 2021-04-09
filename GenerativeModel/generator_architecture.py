# coding=utf-8


"""
Author: zhangjing
Date and time: 2/02/19 - 17:58
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from EmbeddingsImagesDataset import EmbeddingsImagesDataset


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)


class Generator_128(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=4,last_activate='tanh'):
        super(Generator_128, self).__init__()

        nb_channels_input = nb_channels_first_layer * 32

        self.main = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=size_first_layer * size_first_layer * nb_channels_input,
                      bias=False),#4*4*1024
            View(-1, nb_channels_input, size_first_layer, size_first_layer),#1024*4*4
            nn.BatchNorm2d(nb_channels_input, eps=0.001, momentum=0.9),#1024
            nn.ReLU(inplace=True),#1024×4×4

            ConvBlock(nb_channels_input, nb_channels_first_layer * 16, upsampling=True),#512×8×8
            ConvBlock(nb_channels_first_layer * 16, nb_channels_first_layer * 8, upsampling=True),#256×16×16
            ConvBlock(nb_channels_first_layer * 8, nb_channels_first_layer * 4, upsampling=True),#128×32×32
            ConvBlock(nb_channels_first_layer * 4, nb_channels_first_layer * 2, upsampling=True),#64×64×64
            ConvBlock(nb_channels_first_layer * 2, nb_channels_first_layer, upsampling=True),#32×128×128

            ConvBlock(nb_channels_first_layer, nb_channels_output=3, tanh=True,last_activate=last_activate)#3×128×128
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)
class ConvBlock(nn.Module):
    def __init__(self, nb_channels_input, nb_channels_output, upsampling=False, tanh=False,last_activate='tanh'):
        super(ConvBlock, self).__init__()
        self.last_activate = last_activate
        seltorch.tanh = tanh
        self.upsampling = upsampling

        filter_size = 7
        padding = (filter_size - 1) // 2

        if self.upsampling:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(nb_channels_input, nb_channels_output, filter_size, bias=False)
        self.bn_layer = nn.BatchNorm2d(nb_channels_output, eps=0.001, momentum=0.9)

    def forward(self, input_tensor):
        if self.upsampling:
            output = self.up(input_tensor)
        else:
            output = input_tensor

        output = self.pad(output)
        output = self.conv(output)
        output = self.bn_layer(output)

        if seltorch.tanh:
            if(self.last_activate=='tanh'):
                output = torch.tanh(output)
            elif(self.last_activate=='sigmoid'):
                output = F.sigmoid(output)
        else:
            output = F.relu(output)
        return output



class Generator128_res2(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=4,last_activate='tanh'):
        super(Generator128_res2, self).__init__()
        self.last_activate = last_activate
        self.size_first_layer = size_first_layer
        self.nb_channels_input = nb_channels_first_layer * 32#32*32

        self.linear = nn.Sequential(
            nn.Linear(in_features=z_dim,out_features=size_first_layer * size_first_layer * self.nb_channels_input,bias=False),
        )
        self.View = View(-1, self.nb_channels_input, self.size_first_layer, self.size_first_layer)
        self.BN0 = nn.BatchNorm2d(self.nb_channels_input, eps=0.001, momentum=0.9)
        self.Relu0 = nn.ReLU(inplace=True)


        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.nb_channels_input, nb_channels_first_layer*16, 7, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer*16, eps=0.001, momentum=0.9)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(nb_channels_first_layer*16, nb_channels_first_layer*8, 7, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer*8, eps=0.001, momentum=0.9)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(nb_channels_first_layer*8, nb_channels_first_layer*4, 7, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer*4, eps=0.001, momentum=0.9)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(nb_channels_first_layer*4, nb_channels_first_layer*2, 7, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer*2, eps=0.001, momentum=0.9)
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(nb_channels_first_layer*2, nb_channels_first_layer, 7, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer, eps=0.001, momentum=0.9)
        )

        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nb_channels_first_layer, 3, 7, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),
        )

        self.skip_layer1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners = True),
            nn.Conv2d(self.nb_channels_input, nb_channels_first_layer * 8, 1, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer * 8, eps=0.001, momentum=0.9)
        )
        self.skip_layer2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners = True),
            nn.Conv2d(nb_channels_first_layer * 8, nb_channels_first_layer * 2, 1, bias=False),
            nn.BatchNorm2d(nb_channels_first_layer * 2, eps=0.001, momentum=0.9)
        )

    def forward(self, input_tensor):
        input_tensor = self.linear(input_tensor)
        input_tensor = self.View(input_tensor)
        input_tensor = self.Relu0(self.BN0(input_tensor))

        # skip1--2layers
        x1 = self.ConvBlock1(input_tensor)
        x2 = self.ConvBlock2(F.relu(x1))
        x2 = F.relu(x2 + self.skip_layer1(input_tensor))

        # skip2--2layers
        x3 = self.ConvBlock3(x2)
        x4 = self.ConvBlock4(F.relu(x3))
        x4 = F.relu(x4 + self.skip_layer2(x2))

        x5 = self.ConvBlock5(x4)
        if(self.last_activate == 'tanh'):
            output = torch.tanh(self.out_layer(x5))
        elif(self.last_activate == 'sigmoid'):
            output = F.sigmoid(self.out_layer(x5))
        else:
            output = x5
            print("最后一激活函数设置不正确...请检查")
            exit(0)
        return output

class Generator128_mpca(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=8,last_activate='tanh'):
        super(Generator128_mpca, self).__init__()
        self.last_activate = last_activate
        self.size_first_layer = size_first_layer
        self.nb_channels_input = nb_channels_first_layer

        self.ConvBlock0 = nn.Sequential(
            nn.Conv2d(self.nb_channels_input, 512, 1, bias=False),
            nn.BatchNorm2d(512, eps=0.001, momentum=0.9)
        )

        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(512, 256, 7, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(256, 128, 7, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, 64, 7, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 32, 7, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9)
        )


        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, 7, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),
        )



    def forward(self, input_tensor):
        x0 = self.ConvBlock0(input_tensor)
        x0 = F.relu(x0)

        x1 = self.ConvBlock1(x0)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        x4 = self.ConvBlock4(x3)
        x4 = F.relu(x4)

        if(self.last_activate == 'tanh'):
            output = torch.tanh(self.out_layer(x4))
        elif(self.last_activate == 'sigmoid'):
            output = F.sigmoid(self.out_layer(x4))
        else:
            output = x4
            print("最后一激活函数设置不正确...请检查")
            exit(0)
        return output


class Generator32_1(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=2,last_activate='tanh'):
        super(Generator32_1, self).__init__()
        self.last_activate = last_activate
        self.size_first_layer = size_first_layer
        self.nb_channels_input = nb_channels_first_layer * 32 # 32*32

        self.linear = nn.Sequential(
            nn.Linear(in_features=z_dim,out_features=size_first_layer * self.nb_channels_input,bias=False),
        )
        self.View = View(-1, 512, 2, 2)#512*2*2
        self.BN0 = nn.BatchNorm2d(512, eps=0.001, momentum=0.9)
        self.Relu0 = nn.ReLU(inplace=True)


        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(512, 256, 7, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(256, 128, 7, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, 64, 7, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 32, 7, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9)
        )
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, 7, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),
        )


    def forward(self, input_tensor):
        input_tensor = self.linear(input_tensor)
        input_tensor = self.View(input_tensor)
        input_tensor = self.Relu0(self.BN0(input_tensor))

        x1  = self.ConvBlock1(input_tensor)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        x4 = self.ConvBlock4(x3)
        x4 = F.relu(x4)

        output = self.out_layer(x4)
        if(self.last_activate=='tanh') :
            output=torch.tanh(output)
        elif(self.last_activate=='sigmoid'):
            output = F.sigmoid(output)
        else:
            print("最后一激活函数设置不正确...请检查")
            exit(0)
        return output



'''
采样大小为512×2×2=2048,pad=1,卷积核为3，上采样为2倍
'''
class Generator32_2(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=2,last_activate='tanh'):
        super(Generator32_2, self).__init__()
        self.last_activate = last_activate
        self.size_first_layer = size_first_layer
        self.nb_channels_input = nb_channels_first_layer * 32#32*32

        self.linear = nn.Sequential(
            nn.Linear(in_features=z_dim,out_features=size_first_layer * self.nb_channels_input,bias=False),
        )
        self.View = View(-1, 512, 2, 2)#512*2*2
        self.BN0 = nn.BatchNorm2d(512, eps=0.001, momentum=0.9)
        self.Relu0 = nn.ReLU(inplace=True)


        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.9)
        )
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),
        )


    def forward(self, input_tensor):
        input_tensor = self.linear(input_tensor)
        input_tensor = self.View(input_tensor)
        input_tensor = self.Relu0(self.BN0(input_tensor))

        x1  = self.ConvBlock1(input_tensor)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        x4 = self.ConvBlock4(x3)
        x4 = F.relu(x4)

        output = self.out_layer(x4)
        if(self.last_activate=='tanh') :
            output=torch.tanh(output)
        elif(self.last_activate=='sigmoid'):
            output = F.sigmoid(output)
        else:
            print("最后一激活函数设置不正确...请检查")
            exit(0)

        return output


'''
采样大小为1024×2×2=4096,pad=1,卷积核为3，上采样为2倍
'''
class Generator32_3(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=2,last_activate='tanh'):
        super(Generator32_3, self).__init__()
        self.last_activate = last_activate
        self.size_first_layer = size_first_layer
        self.nb_channels_input = nb_channels_first_layer * 32#32*32

        self.linear = nn.Sequential(
            nn.Linear(in_features=z_dim,out_features=self.nb_channels_input*size_first_layer *size_first_layer ,bias=False),
        )
        self.View = View(-1, 1024, 2, 2)#1024*2*2
        self.BN0 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.9)
        self.Relu0 = nn.ReLU(inplace=True)


        self.ConvBlock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 512, 3, bias=False),
            nn.BatchNorm2d(512, eps=0.001, momentum=0.9)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512,256, 3, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.9)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,128, 3, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.9)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        )
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.9),
        )


    def forward(self, input_tensor):
        input_tensor = self.linear(input_tensor)
        input_tensor = self.View(input_tensor)
        input_tensor = self.Relu0(self.BN0(input_tensor))

        x1  = self.ConvBlock1(input_tensor)
        x1 = F.relu(x1)

        x2 = self.ConvBlock2(x1)
        x2 = F.relu(x2)

        x3 = self.ConvBlock3(x2)
        x3 = F.relu(x3)

        x4 = self.ConvBlock4(x3)
        x4 = F.relu(x4)

        output = self.out_layer(x4)
        if(self.last_activate=='tanh') :
            output=torch.tanh(output)
        elif(self.last_activate=='sigmoid'):
            output = F.sigmoid(output)
        else:
            print("最后一激活函数设置不正确...请检查")
            exit(0)

        return output


def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

