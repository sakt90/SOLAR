import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.slimmable_ops import SlimmableConv2d, SlimmableLinear, width_mult_list, SwitchableBatchNorm2d
from models.FiLM import SlimmableFiLM_Layer
from models.DualBN import SwitchableDualBN2d

class SlimmableWideBasicBlockOAT(nn.Module):
    def __init__(self, in_planes_list, out_planes_list, stride, dropRate=0.0, use2BN=True, FiLM_in_channels=1):
        super(SlimmableWideBasicBlockOAT, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = SwitchableDualBN2d
        else:
            Norm2d = SwitchableBatchNorm2d

        self.bn1 = Norm2d(in_planes_list)
        self.conv1 = SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = Norm2d(out_planes_list)
        self.conv2 = SlimmableConv2d(out_planes_list, out_planes_list, kernel_size=3, stride=1, padding=1, bias=False)

        self.droprate = dropRate

        self.equalInOut = (list(in_planes_list) == list(out_planes_list))
        self.convShortcut = (not self.equalInOut) and SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=1, stride=stride, padding=0, bias=False) or None

        self.film1 = SlimmableFiLM_Layer(channels_list=in_planes_list, in_channels=FiLM_in_channels) 
        self.film2 = SlimmableFiLM_Layer(channels_list=out_planes_list, in_channels=FiLM_in_channels)

    def forward(self, x, _lambda, idx2BN=None):
        if self.use2BN:
            out = self.bn1(x, idx2BN)
        else:
            out = self.bn1(x)
        out = self.film1(out, _lambda)
        out = F.relu(out)
        if not self.equalInOut:
            sc = self.convShortcut(out)
        else:
            sc = x
        out = self.conv1(out)
        
        if self.use2BN:
            out = self.bn2(out, idx2BN)
        else:
            out = self.bn2(out)
        out = self.film2(out, _lambda)
        out = F.relu(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        out = torch.add(sc, out)
        return out

class SlimmableWideResNet_40_2_OAT(nn.Module):
    def __init__(self, depth=40, num_classes=10, widen_factor=2, dropRate=0.0, FiLM_in_channels=1, use2BN=True):
        super(SlimmableWideResNet_40_2_OAT, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        # nChannels = [16, 128, 256, 512]
        
        nChannels_0_list = np.array([int(nChannels[0] * width_mult) for width_mult in width_mult_list])
        nChannels_1_list = np.array([int(nChannels[1] * width_mult) for width_mult in width_mult_list])
        nChannels_2_list = np.array([int(nChannels[2] * width_mult) for width_mult in width_mult_list])
        nChannels_3_list = np.array([int(nChannels[3] * width_mult) for width_mult in width_mult_list])
              
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        # n = 2
        
        block = SlimmableWideBasicBlockOAT
        self.use2BN = use2BN

        # 1st conv before any network block
        self.conv1 = SlimmableConv2d([3 for _ in width_mult_list], nChannels_0_list, kernel_size=3, stride=1, padding=1, bias=False)

        ######## 1st block
        self.bundle1 = [block(nChannels_0_list, nChannels_1_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels)]
        for _ in range(1, n):
            self.bundle1.append(block(nChannels_1_list, nChannels_1_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels))
        self.bundle1 = nn.ModuleList(self.bundle1)
        
        ######## 2nd block
        self.bundle2 = [block(nChannels_1_list, nChannels_2_list, 2, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels)]
        for _ in range(1, n):
            self.bundle2.append(block(nChannels_2_list, nChannels_2_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels))
        self.bundle2 = nn.ModuleList(self.bundle2)
        
        ######## 3rd block
        self.bundle3 = [block(nChannels_2_list, nChannels_3_list, 2, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels)]
        for _ in range(1, n):
            self.bundle3.append(block(nChannels_3_list, nChannels_3_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels))
        self.bundle3 = nn.ModuleList(self.bundle3)
        
        ######## global average pooling and classifier
        if self.use2BN:
            self.bn1 = SwitchableDualBN2d(nChannels_3_list)
        else:
            self.bn1 = SwitchableBatchNorm2d(nChannels_3_list)

        self.fc = SlimmableLinear(nChannels_3_list, [num_classes for width_mult in width_mult_list])
        self.bundles = [self.bundle1, self.bundle2, self.bundle3]


    def forward(self, x, _lambda, idx2BN=None):
        out = self.conv1(x)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out, _lambda, idx2BN)
        if self.use2BN:
            out = self.bn1(out, idx2BN) 
        else:
            out = self.bn1(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 24)
        
        out = out.view(out.size(0), -1) ### from Slimmable OAT ResNet34 
        return self.fc(out)
    
####################################################################################################################

class SlimmableWideBasicBlockOAT_SOL(nn.Module):
    def __init__(self, in_planes_list, out_planes_list, stride, dropRate=0.0, use2BN=True, FiLM_in_channels=1):
        super(SlimmableWideBasicBlockOAT_SOL, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = SwitchableDualBN2d
        else:
            Norm2d = SwitchableBatchNorm2d

        self.bn1 = Norm2d(in_planes_list)
        self.conv1 = SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = Norm2d(out_planes_list)
        self.conv2 = SlimmableConv2d(out_planes_list, out_planes_list, kernel_size=3, stride=1, padding=1, bias=False)

        self.droprate = dropRate

        self.equalInOut = (list(in_planes_list) == list(out_planes_list))
        self.convShortcut = (not self.equalInOut) and SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=1, stride=stride, padding=0, bias=False) or None

        self.film1 = SlimmableFiLM_Layer(channels_list=in_planes_list, in_channels=FiLM_in_channels) 
        self.film2 = SlimmableFiLM_Layer(channels_list=out_planes_list, in_channels=FiLM_in_channels)

    def forward(self, x, _lambda, idx2BN=None):
        if self.use2BN:
            out = self.bn1(x, idx2BN)
        else:
            out = self.bn1(x)
        out = self.film1(out, _lambda)
        out = F.relu(out)
        if not self.equalInOut:
            sc = self.convShortcut(out)
        else:
            sc = x
        out = self.conv1(out)
        
        if self.use2BN:
            out = self.bn2(out, idx2BN)
        else:
            out = self.bn2(out)
        out = self.film2(out, _lambda)
        out = F.relu(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        out = torch.add(sc, out)
        return out
    
class SlimmableWideResNet_40_2_OAT_SOL(nn.Module):
    def __init__(self, depth=40, num_classes=10, widen_factor=2, dropRate=0.0, FiLM_in_channels=1, use2BN=True):
        super(SlimmableWideResNet_40_2_OAT_SOL, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        # nChannels = [16, 128, 256, 512]
        
        nChannels_0_list = np.array([int(nChannels[0] * width_mult) for width_mult in width_mult_list])
        nChannels_1_list = np.array([int(nChannels[1] * width_mult) for width_mult in width_mult_list])
        nChannels_2_list = np.array([int(nChannels[2] * width_mult) for width_mult in width_mult_list])
        nChannels_3_list = np.array([int(nChannels[3] * width_mult) for width_mult in width_mult_list])
              
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        # n = 2
        
        block = SlimmableWideBasicBlockOAT_SOL
        self.use2BN = use2BN

        # 1st conv before any network block
        self.conv1 = SlimmableConv2d([3 for _ in width_mult_list], nChannels_0_list, kernel_size=3, stride=1, padding=1, bias=False)

        ######## 1st block
        self.bundle1 = [block(nChannels_0_list, nChannels_1_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels)]
        for _ in range(1, n):
            self.bundle1.append(block(nChannels_1_list, nChannels_1_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels))
        self.bundle1 = nn.ModuleList(self.bundle1)
        
        ######## 2nd block
        self.bundle2 = [block(nChannels_1_list, nChannels_2_list, 2, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels)]
        for _ in range(1, n):
            self.bundle2.append(block(nChannels_2_list, nChannels_2_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels))
        self.bundle2 = nn.ModuleList(self.bundle2)
        
        ######## 3rd block
        self.bundle3 = [block(nChannels_2_list, nChannels_3_list, 2, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels)]
        for _ in range(1, n):
            self.bundle3.append(block(nChannels_3_list, nChannels_3_list, 1, dropRate=dropRate, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels))
        self.bundle3 = nn.ModuleList(self.bundle3)
        
        ######## global average pooling and classifier
        if self.use2BN:
            self.bn1 = SwitchableDualBN2d(nChannels_3_list)
        else:
            self.bn1 = SwitchableBatchNorm2d(nChannels_3_list)

        sub_net_widths = []
        for width_factor in width_mult_list:
            sub_net_widths.append(int(128 * width_factor))
        sub_net_widths

        ####### Separate Output Heads
        self.output_heads = nn.ModuleDict({str(widths): nn.Linear(int(widths), num_classes) for widths in sub_net_widths})
        self.bundles = [self.bundle1, self.bundle2, self.bundle3]

    def forward(self, x, _lambda, idx2BN=None):
        out = self.conv1(x)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out, _lambda, idx2BN)
        if self.use2BN:
            out = self.bn1(out, idx2BN) 
        else:
            out = self.bn1(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 24)
        
        out = out.view(out.size(0), -1) ### from Slimmable OAT ResNet34
        out = self.output_heads[str(out.shape[1])](out)
        return out