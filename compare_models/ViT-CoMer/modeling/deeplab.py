import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from modeling.rfb import build_rfb
# from modeling.aspp_det import build_aspp
from modeling.aspp_dcn import build_aspp
# from modeling.transformer_yuan import build_transformer
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.upernet import UPerHead


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )
        self._init_weight()
    
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


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, aux=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.aux_classifier = None
        if aux:
            if backbone == 'mix_transformer':
                aux_inplanes = 768
            elif backbone == 'deit':
                aux_inplanes = 384
            self.aux_classifier = FCNHead(aux_inplanes, num_classes)
        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, aux)
        # self.rfb = build_rfb(backbone, output_stride, BatchNorm)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # self.transformer = build_transformer(num_classes, BatchNorm)
        # self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.upernet = UPerHead(in_channels=[384, 384, 384, 384],
                     in_index=[0, 1, 2, 3],
                     pool_scales=(1, 2, 3, 6),
                     channels=512,
                     dropout_ratio=0.1,
                     num_classes=2,
                     align_corners=False)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        output = {}
        # x = [2048,32, 32]; low_level_feat = [256, 128, 128]; middle_level_feat = [512, 64, 64]
        x = self.backbone(input)
        # x1 = [256, 32, 32]
        # x1 = self.rfb(x)
        '''
        if self.aux_classifier is not None or isinstance(x, dict):
            x1 = self.aspp(x["out"])
        else:
            x1 = self.aspp(x)
        '''
        # x2 = [512, 16, 16]
        #   x2 = self.transformer(x)
        # x2 = [512, 32, 32]
        #   x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        #   x = self.decoder(x1, x2, low_level_feat)
        output["out"] = self.upernet(x["out"])
        output["out"] = F.interpolate(output["out"], size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.aux_classifier is not None:
            x["aux"] = self.aux_classifier(x["aux"])
            # 使用双线性插值还原回原图尺度
            x["aux"] = F.interpolate(x["aux"], size=input.size()[2:], mode='bilinear', align_corners=True)
            output["aux"] = x["aux"]

        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        if self.aux_classifier is not None:
            modules = [self.upernet, self.aux_classifier]
        else:    
            modules = [self.upernet]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


