import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


# 参考代码：https://github.com/GOATmessi7/ASFF/blob/master/models/network_blocks.py
class ASFF(nn.Module):

    def __init__(self):
        super(ASFF, self).__init__()
        self.dim = [64, 256, 512]
        self.level_0_rechannel = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU6(inplace=True))

        self.level_2_rechannel = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU6(inplace=True))

        self.weight_level = nn.Sequential(
                nn.Conv2d(256, 16, kernel_size=1, stride=1),
                nn.BatchNorm2d(16),
                nn.ReLU6(inplace=True))

        self.weight_levels = nn.Conv2d(16 * 3, 3, kernel_size=1, stride=1, padding=0)

        self.expand = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU6(inplace=True))

    def forward(self, x_level_0, x_level_1, x_level_2):
        level_0_output = self.level_0_rechannel(x_level_0)
        level_1_output = x_level_1
        level_2_compress = self.level_2_rechannel(x_level_2)
        level_2_output = F.interpolate(level_2_compress, size=(128, 128), mode='nearest')

        level_0_weight = self.weight_level(level_0_output)
        level_1_weight = self.weight_level(level_1_output)
        level_2_weight = self.weight_level(level_2_output)
        levels_weight = torch.cat((level_0_weight, level_1_weight, level_2_weight), 1)
        levels_weight = self.weight_levels(levels_weight)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out = level_0_output * levels_weight[:, 0:1, :, :] + \
                            level_1_output * levels_weight[:, 1:2, :, :] + \
                            level_2_output * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    # 这里相对于ResNet，在代码中增加一下两个参数groups和width_per_group（即为group数和conv2中组卷积每个group的卷积核个数）
    # 默认值就是正常的ResNet
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, dilation=1, BatchNorm=None):
        super(Bottleneck, self).__init__()
        # 这里也可以自动计算中间的通道数，也就是3x3卷积后的通道数，如果不改变就是out_channels
        # 如果groups=32,with_per_group=4,out_channels就翻倍了
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm(width)
        # -----------------------------------------
        # 组卷积的数，需要传入参数
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = BatchNorm(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm(out_channel * self.expansion)
        # -----------------------------------------

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Conv2d(out_channel * self.expansion, out_channel * self.expansion // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channel * self.expansion // 16, out_channel * self.expansion, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        x2 = self.se(out)
        out = out * x2

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self,
                 block,  # 表示block的类型
                 blocks_num,  # 表示的是每一层block的个数
                 groups,  # 表示组卷积的数
                 width_per_group,
                 output_stride,
                 BatchNorm,
                 pretrained=True):
        super(ResNeXt, self).__init__()
        self.in_channel = 64
        blocks = [1, 2, 4]

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = BatchNorm(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.asff = ASFF()

        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)  # 64 -> 256
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)  # 256 -> 512
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)  # 512 -> 1024
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],BatchNorm=BatchNorm)   # 1024 ->2048
        # self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 512 -> 1024
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    # 形成单个Stage的网络结构
    def _make_layer(self, block, channel, block_num, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(channel * block.expansion))
        # 该部分是将每个blocks的第一个残差结构保存在layers列表中。
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            dilation=dilation,
                            width_per_group=self.width_per_group,
                            BatchNorm=BatchNorm))
        self.in_channel = channel * block.expansion  # 得到最后的输出

        # 该部分是将每个blocks的剩下残差结构保存在layers列表中，这样就完成了一个blocks的构造。
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                dilation=dilation,
                                BatchNorm=BatchNorm))

        # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.in_channel != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, planes, stride, dilation=blocks[0]*dilation, groups=self.groups,
                            width_per_group=self.width_per_group, downsample=downsample, BatchNorm=BatchNorm))
        self.in_channel = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.in_channel, planes, stride=1, groups=self.groups,
                            width_per_group=self.width_per_group, dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)     # x = [64, 128, 128]

        x2 = self.layer1(x1)      # x = [256, 128, 128]
        # low_level_feat = x2
        x3 = self.layer2(x2)      # x = [512, 64, 64]
        low_level_feat = self.asff(x1, x2, x3)
        x = self.layer3(x3)      # x = [1024, 32, 32]
        x = self.layer4(x)      # x = [2048, 32, 32]

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNeXt101_32x8d(output_stride, BatchNorm, pretrained=True):
    groups = 32
    width_per_group = 8
    return ResNeXt(Bottleneck,
                   [3, 4, 23, 3],
                   groups=groups,
                   width_per_group=width_per_group,
                   output_stride=output_stride,
                   BatchNorm=BatchNorm, pretrained=pretrained)

if __name__ == "__main__":
    import torch
    model = ResNeXt101_32x8d(BatchNorm=nn.BatchNorm2d, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())