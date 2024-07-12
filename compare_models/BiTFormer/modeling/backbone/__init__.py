from modeling.backbone import resnet, xception, drn, mobilenet, mix_transformer_yuan, swin_transformer, convnextv2, bitformer

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'mix_transformer':
        return mix_transformer_yuan.build_transformer()
    elif backbone == 'swin_transformer':
        return swin_transformer.build_transformer()
    elif backbone == 'convnext':
        return convnextv2.convnextv2_base()
    elif backbone == 'bitformer':
        return bitformer.biformer_base()
    else:
        raise NotImplementedError
