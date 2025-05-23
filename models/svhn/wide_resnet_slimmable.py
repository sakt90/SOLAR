"""PyTorch implementation of Wide-ResNet taken from 
https://github.com/jeromerony/fast_adversarial/blob/master/fast_adv/models/cifar10/wide_resnet.py"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.slimmable_ops import SlimmableConv2d, SlimmableLinear, width_mult_list, SwitchableBatchNorm2d

class Slimmable_Wide_BasicBlock(nn.Module):
    def __init__(self, in_planes_list, out_planes_list, stride, dropRate=0.0):  ####################### Lists Issue
        super(Slimmable_Wide_BasicBlock, self).__init__()
        self.bn1 = SwitchableBatchNorm2d(in_planes_list)
        self.conv1 = SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(out_planes_list)
        self.conv2 = SlimmableConv2d(out_planes_list, out_planes_list, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (list(in_planes_list) == list(out_planes_list)) ############## Issue
        self.convShortcut = (not self.equalInOut) and SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x), inplace=True)
        else:
            out = F.relu(self.bn1(x), inplace=True)
        out = F.relu(self.bn2(self.conv1(out if self.equalInOut else x)), inplace=True)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes_list, out_planes_list, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes_list, out_planes_list, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes_list, out_planes_list, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes_list or out_planes_list, out_planes_list, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class SlimmableWideResNet_16_8(nn.Module):
    def __init__(self, depth=16, num_classes=10, widen_factor=8, dropRate=0.0):
        super(SlimmableWideResNet_16_8, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = Slimmable_Wide_BasicBlock
        
        nChannels_0_list = np.array([int(nChannels[0] * width_mult) for width_mult in width_mult_list])
        nChannels_1_list = np.array([int(nChannels[1] * width_mult) for width_mult in width_mult_list])
        nChannels_2_list = np.array([int(nChannels[2] * width_mult) for width_mult in width_mult_list])
        nChannels_3_list = np.array([int(nChannels[3] * width_mult) for width_mult in width_mult_list])
        
        # 1st conv before any network block
        self.conv1 = SlimmableConv2d([3 for _ in width_mult_list], nChannels_0_list, kernel_size=3, stride=1, padding=1, bias=False)
               
        ########
        self.bundle1 = [block(nChannels_0_list, nChannels_1_list, 1, dropRate=dropRate)]
        for _ in range(1, n):
            self.bundle1.append(block(nChannels_1_list, nChannels_1_list, 1, dropRate=dropRate))
        self.bundle1 = nn.ModuleList(self.bundle1)
        
        self.bundle2 = [block(nChannels_1_list, nChannels_2_list, 2, dropRate=dropRate)]
        for _ in range(1, n):
            self.bundle2.append(block(nChannels_2_list, nChannels_2_list, 1, dropRate=dropRate))
        self.bundle2 = nn.ModuleList(self.bundle2)

        self.bundle3 = [block(nChannels_2_list, nChannels_3_list, 2, dropRate=dropRate)]
        for _ in range(1, n):
            self.bundle3.append(block(nChannels_3_list, nChannels_3_list, 1, dropRate=dropRate))
        self.bundle3 = nn.ModuleList(self.bundle3)
        
        self.bundles = [self.bundle1, self.bundle2, self.bundle3]
        
        # global average pooling and classifier
        self.bn1 = SwitchableBatchNorm2d(nChannels_3_list)
        self.fc = SlimmableLinear(nChannels_3_list, [num_classes for width_mult in width_mult_list])

    def forward(self, x):
        out = self.conv1(x)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out)
     
        out = F.relu(self.bn1(out), inplace=True)
        out = F.avg_pool2d(out, 8)
        
        out = out.view(out.size(0), -1)
        return self.fc(out)

class SlimmableWideResNet_16_8_SOL(nn.Module):
    def __init__(self, depth=16, num_classes=10, widen_factor=8, dropRate=0.0):
        super(SlimmableWideResNet_16_8_SOL, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = Slimmable_Wide_BasicBlock
        
        nChannels_0_list = np.array([int(nChannels[0] * width_mult) for width_mult in width_mult_list])
        nChannels_1_list = np.array([int(nChannels[1] * width_mult) for width_mult in width_mult_list])
        nChannels_2_list = np.array([int(nChannels[2] * width_mult) for width_mult in width_mult_list])
        nChannels_3_list = np.array([int(nChannels[3] * width_mult) for width_mult in width_mult_list])
        
        # 1st conv before any network block
        self.conv1 = SlimmableConv2d([3 for _ in width_mult_list], nChannels_0_list, kernel_size=3, stride=1, padding=1, bias=False)
        
        ########
        self.bundle1 = [block(nChannels_0_list, nChannels_1_list, 1, dropRate=dropRate)]
        for _ in range(1, n):
            self.bundle1.append(block(nChannels_1_list, nChannels_1_list, 1, dropRate=dropRate))
        self.bundle1 = nn.ModuleList(self.bundle1)
        
        self.bundle2 = [block(nChannels_1_list, nChannels_2_list, 2, dropRate=dropRate)]
        for _ in range(1, n):
            self.bundle2.append(block(nChannels_2_list, nChannels_2_list, 1, dropRate=dropRate))
        self.bundle2 = nn.ModuleList(self.bundle2)

        self.bundle3 = [block(nChannels_2_list, nChannels_3_list, 2, dropRate=dropRate)]
        for _ in range(1, n):
            self.bundle3.append(block(nChannels_3_list, nChannels_3_list, 1, dropRate=dropRate))
        self.bundle3 = nn.ModuleList(self.bundle3)
        
        self.bundles = [self.bundle1, self.bundle2, self.bundle3]
        
        # global average pooling and classifier
        self.bn1 = SwitchableBatchNorm2d(nChannels_3_list)
        sub_net_widths = []
        for width_factor in width_mult_list:
            sub_net_widths.append(int(512 * width_factor))
        sub_net_widths

        ####### Separate Output Heads
        self.output_heads = nn.ModuleDict({str(widths): nn.Linear(int(widths), num_classes) for widths in sub_net_widths})

    def forward(self, x):
        out = self.conv1(x)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out)
     
        out = F.relu(self.bn1(out), inplace=True)
        out = F.avg_pool2d(out, 8)

        out = out.view(out.size(0), -1)
        out = self.output_heads[str(out.shape[1])](out)
        return out