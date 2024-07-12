import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.ops_dcnv3.modules.dcnv3 import DCNv3_pytorch, build_act_layer, build_norm_layer


# -----------------drop path-------------------#
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
# -----------------drop path-------------------#


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _DCNASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_DCNASPPModule, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                                  BatchNorm(planes),
                                  nn.LeakyReLU())
        self.dcn = DCNv3_pytorch(
            channels=planes,  # 通道数
            kernel_size=kernel_size,  # 卷积核大小
            dw_kernel_size=None,  # 深度可分离卷积核大小
            stride=1,  # 步长
            pad=padding,  # 填充
            dilation=dilation,  # 空洞率
            group=4,  # 分组数
            offset_scale=1.0,
            act_layer='LeakyReLU',  # 激活函数
            norm_layer='BN',  # 归一化层
            center_feature_scale=False)
        self.mlp = MLPLayer(in_features=planes, hidden_features=int(planes * 4.0), act_layer='LeakyReLU', drop=0.)
        self.bn = build_norm_layer(planes, 'BN')
        self.se = SEBlock(planes, planes // 4)
        self.drop_path = DropPath(0.2)

    def forward(self, input):
        input = self.conv(input)
        input = input.permute(0, 2, 3, 1)
        x = input + self.drop_path(self.dcn(input))
        x = self.bn(x)
        
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        x = x.permute(0, 2, 3, 1)
        
        x = self.mlp(x)

        return (input + self.drop_path(x)).permute(0, 3, 1, 2)


class _DSCASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_DSCASPPModule, self).__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.Conv2d(planes, planes, kernel_size=kernel_size,
                      stride=1, padding=padding, groups=planes, dilation=dilation, bias=False)
        )
        self.bn = BatchNorm(planes)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_DCN(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP_DCN, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'mix_transformer':
            inplanes = 768
        elif backbone == 'deit':
            inplanes = 384
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 3, 6, 12, 18, 24]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _DSCASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _DCNASPPModule(inplanes * 2, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _DCNASPPModule(inplanes * 2, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _DCNASPPModule(inplanes * 2, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.aspp5 = _DCNASPPModule(inplanes * 2, 256, 3, padding=dilations[4], dilation=dilations[4], BatchNorm=BatchNorm)
        self.aspp6 = _DCNASPPModule(inplanes * 2, 256, 3, padding=dilations[5], dilation=dilations[5], BatchNorm=BatchNorm)
        
        # self.SP = StripPooling(inplanes, BatchNorm=BatchNorm)
        
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.LeakyReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(256, inplanes, 1, stride=1, bias=False),
                                             BatchNorm(inplanes),
                                             nn.LeakyReLU())
        
        self.conv1 = nn.Conv2d(256 * 7, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        
        x1 = self.aspp1(x)
        x_residual = self.conv2(x1)
        x2 = self.aspp2(torch.cat((x, x_residual), dim=1))
        x_residual = self.conv2(x2)
        x3 = self.aspp3(torch.cat((x, x_residual), dim=1))
        x_residual = self.conv2(x3)
        x4 = self.aspp4(torch.cat((x, x_residual), dim=1))
        x_residual = self.conv2(x4)
        x5 = self.aspp5(torch.cat((x, x_residual), dim=1))
        x_residual = self.conv2(x5)
        x6 = self.aspp6(torch.cat((x, x_residual), dim=1))

        x7 = self.global_avg_pool(x)
        x7 = F.interpolate(x7, size=x6.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)
 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP_DCN(backbone, output_stride, BatchNorm)
    

if __name__ == "__main__":
    input = torch.rand(2, 768, 32, 32)
    model = build_aspp('mix_transformer', 16, nn.BatchNorm2d)
    output = model(input)
    print(output.size())