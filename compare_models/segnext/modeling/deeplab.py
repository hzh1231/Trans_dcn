import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from modeling.rfb import build_rfb
# from modeling.aspp_det import build_aspp
from modeling.aspp import build_aspp
# from modeling.transformer_yuan import build_transformer
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('Number of parameter: % .4fM' % (total / 1e6))
        # self.rfb = build_rfb(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # self.transformer = build_transformer(num_classes, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        # x = [2048,32, 32]; low_level_feat = [256, 128, 128]; middle_level_feat = [512, 64, 64]
        x, low_level_feat = self.backbone(input)
        # x1 = [256, 32, 32]
        # x1 = self.rfb(x)
        # x1 = self.aspp(x)
        # x2 = [512, 16, 16]
        #   x2 = self.transformer(x)
        # x2 = [512, 32, 32]
        #   x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        #   x = self.decoder(x1, x2, low_level_feat)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

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
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


