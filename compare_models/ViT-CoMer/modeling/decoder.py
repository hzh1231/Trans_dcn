import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.cbam import CBAM

        
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


class DSCModule(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation, BatchNorm=None, relu=True):
        super(DSCModule, self).__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
            nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size,
                                            stride=1, padding=padding, groups=outplanes, dilation=dilation, bias=False)
        )
        if BatchNorm is not None:
            self.bn = BatchNorm(outplanes)
        else:
            self.bn = nn.Identity()
        self.relu = nn.LeakyReLU() if relu is True else (relu if isinstance(relu, nn.Module) else nn.Identity())

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
        
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'mix_transformer':
            low_level_inplanes = 128
        elif backbone == 'swin_transformer':
            low_level_inplanes = 192
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.LeakyReLU()
        # self.cbam = CBAM(48)
        
        '''
        self.last_conv = nn.Sequential(nn.Conv2d(256 + 128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        '''
        
        self.last_conv = nn.Sequential(DSCModule(256 + 48, 256, kernel_size=3, padding=1, dilation=1, BatchNorm=BatchNorm, relu=True),
                                       nn.Dropout(0.5),
                                       DSCModule(256, 256, kernel_size=3, padding=1, dilation=1, BatchNorm=BatchNorm, relu=True),
                                       nn.Dropout(0.1),
                                       DSCModule(256, num_classes, kernel_size=1, padding=1, dilation=1, BatchNorm=None, relu=False))
        
        self._init_weight()
    
    
    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        # low_level_feat = self.cbam(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


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

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
