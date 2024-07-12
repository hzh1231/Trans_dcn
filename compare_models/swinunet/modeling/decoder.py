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
            low_level_inplanes = 768
        elif backbone == 'swin_transformer':
            low_level_inplanes = 192
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.cbam1 = CBAM(48)
        
        self.conv2 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn2 = BatchNorm(48)
        self.cbam2 = CBAM(48)
        
        '''
        self.mid_conv = nn.Sequential(nn.Conv2d(256 + 48, 256 + 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256 + 48),
                                       nn.ReLU())
        '''
        self.mid_conv = ECANet(256 + 48)

        self.last_conv = nn.Sequential(nn.Conv2d(256 + 96, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self._init_weight()
    
    
    def forward(self, x, low_level_feat, mid_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        low_level_feat = self.cbam1(low_level_feat)
        
        mid_level_feat = self.conv2(mid_level_feat)
        mid_level_feat = self.bn2(mid_level_feat)
        mid_level_feat = self.relu(mid_level_feat)
        # mid_level_feat = self.cbam2(mid_level_feat)
        
        x = F.interpolate(x, size=mid_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, mid_level_feat), dim=1)
        x = self.mid_conv(x)
        

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
