#%%

import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from modeling.bricks import DownSample, LayerScale, StochasticDepth, DWConv3x3, DWConv

class StemConv(nn.Module):
    '''following ConvNext paper'''
    def __init__(self, in_channels, out_channels, bn_momentum=0.99):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels//2,
                                                kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                    nn.BatchNorm2d(out_channels//2, eps=1e-5, momentum=float(0.9)),
                                    nn.GELU(),
                                    nn.Conv2d(out_channels//2, out_channels,
                                                kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                    nn.BatchNorm2d(out_channels, eps=1e-5, momentum=float(0.9))
                                )
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        return x, H, W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        skip = x.clone()
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x + skip


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.d_model = dim
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 ls_init_val=0.):
        super(Block, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5, momentum=float(0.9))
        self.attn = SpatialAttention(dim)
        self.drop_path = StochasticDepth(p=drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5, momentum=float(0.9))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = ls_init_val
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class backbone(nn.Module):
    def __init__(self, in_channnels=3, embed_dims=[32, 64, 460,256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2], num_stages=4,
                 ls_init_val=1e-2, drop_path=0.0):
        super(backbone, self).__init__()
        self.backbone = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               drop_path=drop_path)
    def forward(self,x):
        x = self.backbone(x)
        return x

class MSCANet(nn.Module):
    def __init__(self, in_channnels=3, embed_dims=[32, 64, 460,256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2], num_stages=4,
                 ls_init_val=1e-2, drop_path=0.0):
        super(MSCANet, self).__init__()
        # print(f'MSCANet {drop_path}')
        self.depths = depths
        self.num_stages = num_stages
        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_channnels, embed_dims[0])
            else:
                patch_embed = DownSample(in_channels=embed_dims[i-1], embed_dim=embed_dims[i])
            
            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=ffn_ratios[i],
                                   ls_init_val=ls_init_val, drop_path=dpr[cur + j])
                                   for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i+1}', patch_embed)
            setattr(self, f'block{i+1}', block)
            setattr(self, f'norm{i+1}', norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i+1}')
            block = getattr(self, f'block{i+1}')
            norm = getattr(self, f'norm{i+1}')
            
            x, H, W = patch_embed(x)
            
            for blk in block:
                x = blk(x,H,W)
            
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


#%%
# from torchsummary import summary
# model = MSCANet(in_channnels=3, embed_dims=[32, 64, 460,256],
#                  ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2],
#                  num_stages = 4, ls_init_val=1e-2, drop_path=0.0)
# # summary(model, (3,1024,2048))


# y = torch.randn((6,3,1024,2048))#.to('cuda' if torch.cuda.is_available() else 'cpu')
# x = model.forward(y)

# for i in range(4):
#     print(x[i].shape)
# %%
# output shoudl be something like

# torch.Size([6, 32, 256, 512])
# torch.Size([6, 64, 128, 256])
# torch.Size([6, 460, 64, 128])
# torch.Size([6, 256, 32, 64])