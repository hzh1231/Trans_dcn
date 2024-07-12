import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


'''
class StripPooling(nn.Module):
    def __init__(self, in_channels, BatchNorm=None, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   BatchNorm(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   BatchNorm(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   BatchNorm(in_channels))
        self._up_kwargs = up_kwargs
        self._init_weight()

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)
        
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


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(ch, ch//re, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(ch//re, ch, 1),
                                  nn.Sigmoid())
        
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                  nn.Sigmoid())
        
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)                      

class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output     
'''    


class _DSCASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_DSCASPPModule, self).__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.Conv2d(planes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, groups=planes, dilation=dilation, bias=False)
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

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


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'mix_transformer':
            inplanes = 768
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 3, 6, 12, 18, 24]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _DSCASPPModule(inplanes, 512, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _DSCASPPModule(inplanes * 2, 512, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _DSCASPPModule(inplanes * 2, 512, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _DSCASPPModule(inplanes * 2, 512, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.aspp5 = _DSCASPPModule(inplanes * 2, 512, 3, padding=dilations[4], dilation=dilations[4], BatchNorm=BatchNorm)
        self.aspp6 = _DSCASPPModule(inplanes * 2, 512, 3, padding=dilations[5], dilation=dilations[5], BatchNorm=BatchNorm)
        
        # self.SP = StripPooling(inplanes, BatchNorm=BatchNorm)
        
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 512, 1, stride=1, bias=False),
                                             BatchNorm(512),
                                             nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(512, inplanes, 1, stride=1, bias=False),
                                             BatchNorm(inplanes),
                                             nn.ReLU())

        
        self.conv1 = nn.Conv2d(512 * 7, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        
        '''
        x1 = self.aspp1(x)
        x_residual = self.conv2(x1)
        residual = self.relu(x_residual + x)
        x2 = self.aspp3(residual)
        x_residual = self.conv2(x2)
        residual = self.relu(x_residual + x)
        x3 = self.aspp4(residual)
        x_residual = self.conv2(x3)
        residual = self.relu(x_residual + x)
        x4 = self.aspp5(residual)
        x_residual = self.conv2(x4)
        residual = self.relu(x_residual + x)
        x5 = self.aspp5(residual)
        x_residual = self.conv2(x5)
        residual = self.relu(x_residual + x)
        x6 = self.aspp6(residual)
        '''
        
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

        
        # x = self.eca(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

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
    return ASPP(backbone, output_stride, BatchNorm)
