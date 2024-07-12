from modeling.backbone import resnet, xception, drn, mobilenet, mix_transformer, swin_transformer, vit_comer

def build_backbone(backbone, output_stride, BatchNorm, aux):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'mix_transformer':
        return mix_transformer.build_transformer(aux=aux)
    elif backbone == 'swin_transformer':
        return swin_transformer.build_transformer()
    elif backbone == 'deit':
        return vit_comer.build_deit(aux=aux)
    else:
        raise NotImplementedError
