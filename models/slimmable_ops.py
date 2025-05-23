import torch.nn as nn

####################### Select the Required Width Multiplier List for Packing K number of Sub-Nets in the BackBone


# width_mult_list = [1/4, 2/4, 3/4, 4/4]
#width_mult_list = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8]

#width_mult_list = [i/16 for i in range(1, 17)] # 16 Sub-Nets
width_mult_list = [i/32 for i in range(1, 33)] # 32 Sub-Nets


#################################################################################################################################

def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Making sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        assert len(num_features_list) == len(width_mult_list)
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        assert len(in_channels_list) == len(width_mult_list)
        assert len(out_channels_list) == len(width_mult_list)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y

class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        assert len(in_features_list) == len(width_mult_list)
        assert len(out_features_list) == len(width_mult_list)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)