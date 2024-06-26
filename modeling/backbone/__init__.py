from modeling.backbone import resnet, xception, drn, mobilenet, transformer, swin_transformer

def build_backbone(backbone, output_stride, BatchNorm, aux):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'transformer':
        return transformer.build_transformer(aux=aux)
    elif backbone == 'swin_transformer':
        return swin_transformer.build_transformer()
    else:
        raise NotImplementedError
