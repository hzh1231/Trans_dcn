#%%
import math, os
import torch
import torch.nn.functional as F
import torch.nn as nn

from modeling.msca import backbone
from modeling.decoder import decoder_head


class SegNext(nn.Module):
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, config=512, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = backbone(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               drop_path=drop_path)
        self.decoder = decoder_head(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        self.init_weights()

        checkpoint = torch.load('./segnext_base_512x512_ade_160k.pth')
        missing_keys, unexpected_keys = self.encoder.load_state_dict(checkpoint["state_dict"], strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)  # here output will be B x C x H/8 x W/8
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True) #now its same as input
        #  bilinear interpol was used originally
        return output
