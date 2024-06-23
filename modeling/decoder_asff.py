import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

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


class ASFF(nn.Module):

    def __init__(self, dim=[64, 128, 320]):
        super(ASFF, self).__init__()
        self.level_0_rechannel = nn.Sequential(
            nn.Conv2d(dim[0], 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.level_2_rechannel = nn.Sequential(
            nn.Conv2d(dim[2], 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.weight_level = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.weight_levels = nn.Conv2d(16 * 3, 3, kernel_size=1, stride=1, padding=0)

        self.expand = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

    def forward(self, low_level_feat):
        x_level_0, x_level_1, x_level_2 = low_level_feat

        level_0_output = self.level_0_rechannel(x_level_0)
        level_1_output = F.interpolate(x_level_1, size=level_0_output.size()[2:], mode='bilinear', align_corners=False)#upsample
        level_2_output = self.level_2_rechannel(x_level_2)
        level_2_output = F.interpolate(level_2_output, size=level_0_output.size()[2:], mode='bilinear',
                                       align_corners=False)
        level_0_weight = self.weight_level(level_0_output)
        level_1_weight = self.weight_level(level_1_output)
        level_2_weight = self.weight_level(level_2_output)
        levels_weight = torch.cat((level_2_weight, level_1_weight, level_0_weight), 1)
        levels_weight = self.weight_levels(levels_weight)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out = level_0_output * levels_weight[:, 2:, :, :] + \
                    level_1_output * levels_weight[:, 1:2, :, :] + \
                    level_2_output * levels_weight[:, 0:1, :, :]

        out = self.expand(fused_out)

        return out

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'transformer':
            low_level_inplanes = 128
            low_level_channel = [64, 128, 320]
        elif backbone == 'swin_transformer':
            low_level_inplanes = 192
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.LeakyReLU()

        self.asff = ASFF(low_level_channel)

        self.last_conv = nn.Sequential(
            DSCModule(256 + 128, 256, kernel_size=3, padding=1, dilation=1, BatchNorm=BatchNorm, relu=True),
            nn.Dropout(0.5),
            DSCModule(256, 256, kernel_size=3, padding=1, dilation=1, BatchNorm=BatchNorm, relu=True),
            nn.Dropout(0.1),
            DSCModule(256, num_classes, kernel_size=1, padding=1, dilation=1, BatchNorm=None, relu=False))

        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.asff(low_level_feat)
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
    
if __name__ == "__main__":
    import torch
    from thop import profile                            
    from thop import clever_format
    
    a1 = torch.rand(1, 64, 128, 128)
    a2 = torch.rand(1, 128, 64, 64)
    a3 = torch.rand(1, 320, 32, 32)
    
    input = [a1, a2, a3]
    model = ASFF()
    flops, params = profile(model, inputs=(input, ))
    flops,params = clever_format([flops, params],"%.3f")
    print(flops,params)

