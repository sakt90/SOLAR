''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from models.slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d, SlimmableLinear
from models.slimmable_ops import width_mult_list

class SlimmableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes_lst, out_planes_lst, stride=1):
        super(SlimmableBasicBlock, self).__init__()
        self.conv1 = SlimmableConv2d(in_planes_lst, out_planes_lst, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(out_planes_lst)
        self.conv2 = SlimmableConv2d(out_planes_lst, out_planes_lst, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(out_planes_lst)

        self.shortcut = nn.Sequential()
        if stride != 1 or list(in_planes_lst) != list(out_planes_lst):
            self.shortcut = nn.Sequential(
                SlimmableConv2d(in_planes_lst, out_planes_lst, kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(out_planes_lst),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        return out

class SlimmableResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SlimmableResNet, self).__init__()
        self.in_planes_list = np.array([int(64 * width_mult) for width_mult in width_mult_list])

        self.conv1 = SlimmableConv2d(np.array([3 for _ in width_mult_list]), self.in_planes_list,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(self.in_planes_list)
        self.layer1 = self._make_layer(block, np.array([int(64 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, np.array([int(128 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, np.array([int(256 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, np.array([int(512* width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[3], stride=2)
        self.linear = SlimmableLinear(
            np.array([int(512* width_mult) for width_mult in width_mult_list]) * block.expansion, 
            np.array([num_classes for width_mult in width_mult_list])
        )

    def _make_layer(self, block, planes_list, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_list, planes_list, stride))
            self.in_planes_list = planes_list * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#######################################################################################################

class SlimmableResNet_SOL(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SlimmableResNet_SOL, self).__init__()
        self.in_planes_list = np.array([int(64 * width_mult) for width_mult in width_mult_list])

        self.conv1 = SlimmableConv2d(np.array([3 for _ in width_mult_list]), self.in_planes_list,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(self.in_planes_list)
        self.layer1 = self._make_layer(block, np.array([int(64 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, np.array([int(128 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, np.array([int(256 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, np.array([int(512* width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[3], stride=2)
        
        sub_net_widths = []
        for width_factor in width_mult_list:
            sub_net_widths.append(int(512 * width_factor))
        sub_net_widths

        ####### Separate Output Heads
        self.output_heads = nn.ModuleDict({str(widths): nn.Linear(int(widths), num_classes) for widths in sub_net_widths})

    def _make_layer(self, block, planes_list, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_list, planes_list, stride))
            self.in_planes_list = planes_list * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.output_heads[str(out.shape[1])](out)
        return out

#######################################################################################################

def SlimmableResNet34():
    return SlimmableResNet(SlimmableBasicBlock, [3,4,6,3])

def SlimmableResNet34_SOL():
    return SlimmableResNet_SOL(SlimmableBasicBlock, [3,4,6,3])

if __name__ == '__main__':
    model = SlimmableResNet34()
    for width_mult in sorted(width_mult_list, reverse=True):
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        # for name, m in model.named_modules():
        #     if isinstance(m, SlimmableLinear) or isinstance(m, SlimmableConv2d) or isinstance(m, SwitchableBatchNorm2d):
        #         print(name, m.width_mult)