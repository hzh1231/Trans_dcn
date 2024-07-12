import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class BasicConv(nn.Module):
    def __init__(self, in_planes=768, out_planes=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, BatchNorm=nn.BatchNorm2d):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = BatchNorm(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
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


class RFB(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(RFB, self).__init__()

        if backbone == 'drn':
            in_planes = 512
        elif backbone == 'mobilenet':
            in_planes = 320
        elif backbone == 'mix_transformer':
            in_planes = 768
        else:
            in_planes = 2048
        if output_stride == 16:
            pass
        elif output_stride == 8:
            raise NotImplementedError
        else:
            raise NotImplementedError

        out_planes = 256
        inter_planes = in_planes // 4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False, BatchNorm=BatchNorm)
                )

        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0), BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, BatchNorm=BatchNorm)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=1, padding=(0, 1), BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, BatchNorm=BatchNorm)
                )

        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), BatchNorm=BatchNorm),
                BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0), BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False, BatchNorm=BatchNorm)
                )
                
        '''
        self.branch4 = nn.Sequential(
                BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), BatchNorm=BatchNorm),
                BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0), BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=12, dilation=12, relu=False, BatchNorm=BatchNorm)
                )

        self.branch5 = nn.Sequential(
                BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), BatchNorm=BatchNorm),
                BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0), BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=18, dilation=18, relu=False, BatchNorm=BatchNorm)
                )

        self.branch6 = nn.Sequential(
                BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1, BatchNorm=BatchNorm),
                BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), BatchNorm=BatchNorm),
                BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0), BatchNorm=BatchNorm),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=24, dilation=24, relu=False, BatchNorm=BatchNorm)
                )
        '''
        '''
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=1, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2+1, dilation=2+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*2+1, dilation=2*2+1, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*3+1, dilation=2*3+1, relu=False)
                )
        '''
        
        self.conv2 = BasicConv(inter_planes, in_planes, kernel_size=1, stride=1, relu=True, BatchNorm=BatchNorm)
        
        self.ConvLinear = BasicConv(3 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        
        x0 = self.branch0(x)
        residual = self.conv2(x0)
        residual = self.relu(x + residual)
        x1 = self.branch1(residual)
        residual = self.conv2(x1)
        residual = self.relu(x + residual)
        x2 = self.branch2(residual)
        residual = self.conv2(x2)
        residual = self.relu(x + residual)
        x3 = self.branch3(residual)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out + short
        out = self.relu(out)
        x = self.dropout(out)

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


def build_rfb(backbone, output_stride, BatchNorm):
    return RFB(backbone, output_stride, BatchNorm)


if __name__ == "__main__":
    input = torch.rand(1, 768, 32, 32)
    model = build_rfb(backbone='mix_transformer', output_stride=16, BatchNorm=nn.BatchNorm2d)
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))
    output = model(input)
    print(output.size())
