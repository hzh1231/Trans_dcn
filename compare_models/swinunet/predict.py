import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

from networks.swin_unet.vision_transformer import SwinUnet as ViT_seg
from config import get_config


def imgcrop(img):
    w, h = img.size
    x1 = random.randint(0, w - 224)
    y1 = random.randint(0, h - 224)
    img = img.crop((x1, y1, x1 + 224, y1 + 224))
    return img

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default='./test/', help='image to test')
    parser.add_argument('--out-path', type=str, default='./output/', help='mask image to save')
    parser.add_argument('--backbone', type=str, default='mix_transformer',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'mix_transformer'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='./run/pascal/deeplab-mix_transformer/model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes','invoice'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--cfg', type=str,
                    default="configs/swin_tiny_patch4_window7_224_lite.yaml",
                    metavar="FILE", help='path to config file', )
    parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',)
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size for  training (default: auto)')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    config = get_config(args)
    model = ViT_seg(config, img_size=224, num_classes=args.num_classes)
    model.load_from(config)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.RandomScaleCrop(base_size=512, crop_size=args.crop_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    for name in os.listdir(args.in_path):
        s_time = time.time()
        image = Image.open(args.in_path+"/"+name).convert('RGB')

        # image = Image.open(args.in_path).convert('RGB')
        target = Image.open(args.in_path+"/"+name).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                                3, normalize=False, range=(0, 255))
        save_image(grid_image,args.out_path+"/"+"{}_mask.png".format(name[0:-4]))
        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(name,img_time))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in in_path.")


if __name__ == "__main__":
   main()

