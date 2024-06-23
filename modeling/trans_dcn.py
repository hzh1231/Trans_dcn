import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp_dcn import build_aspp
from modeling.decoder_asff import build_decoder
from modeling.backbone import build_backbone


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


class Trans_dcn(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, aux=False):
        super(Trans_dcn, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.aux_classifier = None
        if aux:
            if backbone == 'transformer':
                aux_inplanes = 768
            self.aux_classifier = FCNHead(aux_inplanes, num_classes)
        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, aux)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        output = {}
        x, low_level_feat = self.backbone(input)
        # x1 = [256, 32, 32]
        if self.aux_classifier is not None or isinstance(x, dict):
            x1 = self.aspp(x["out"])
        else:
            x1 = self.aspp(x)
        output["out"] = self.decoder(x1, low_level_feat)
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
            modules = [self.aspp, self.decoder, self.aux_classifier]
        else:    
            modules = [self.aspp, self.decoder]
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
    model = Trans_dcn(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


