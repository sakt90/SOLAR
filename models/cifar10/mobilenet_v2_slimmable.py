import math
import torch.nn as nn
from models.slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d
from models.slimmable_ops import make_divisible
from models.slimmable_ops import width_mult_list

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = [i * expand_ratio for i in inp]
        if expand_ratio != 1:
            layers += [
                SlimmableConv2d(inp, expand_inp, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SlimmableConv2d(
                expand_inp, expand_inp, 3, stride, 1,
                groups_list=expand_inp, bias=False),
            SwitchableBatchNorm2d(expand_inp),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(expand_inp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super(Model, self).__init__()

        self.reset_parameters = False
        
        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  #2
            [6, 32, 3, 1],  #2
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = [
            make_divisible(32 * width_mult)
            for width_mult in width_mult_list]
        self.outp = make_divisible(
            1280 * max(width_mult_list)) if max(
                width_mult_list) > 1.0 else 1280
        first_stride = 1    #2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3,
                    first_stride, 1, bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = [
                make_divisible(c * width_mult)
                for width_mult in width_mult_list]
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(channels, outp, s, t))
                else:
                    self.features.append(
                        InvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    channels,
                    [self.outp for _ in range(len(channels))],
                    1, 1, 0, bias=False),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = 8   #input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        if self.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)        
        x = x.view(-1, self.outp)        
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
##################################### SOL Model ###################################

class Model_SOL(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super(Model_SOL, self).__init__()

        self.reset_parameters = False
        
        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  #2
            [6, 32, 3, 1],  #2
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = [
            make_divisible(32 * width_mult)
            for width_mult in width_mult_list]
        self.outp = make_divisible(
            1280 * max(width_mult_list)) if max(
                width_mult_list) > 1.0 else 1280
        first_stride = 1    #2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3,
                    first_stride, 1, bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = [
                make_divisible(c * width_mult)
                for width_mult in width_mult_list]
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(channels, outp, s, t))
                else:
                    self.features.append(
                        InvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    channels,
                    [self.outp for _ in range(len(channels))],
                    1, 1, 0, bias=False),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = 8   #input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        ################################## classifier
        #self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        
        sub_net_widths = []
        for width_factor in width_mult_list:
            sub_net_widths.append(int(self.outp * width_factor))
        sub_net_widths

        ####### Separate Output Heads
        self.output_heads = nn.ModuleDict({str(widths): nn.Linear(int(widths), num_classes) for widths in sub_net_widths})
        
        if self.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.output_heads[str(x.shape[1])](x)
        #x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
###############################################################################################

def SlimmableMobileNetV2():
    return Model()

def SlimmableMobileNetV2_SOL():
    return Model_SOL()