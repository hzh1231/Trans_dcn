import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image
from utils.pytorch_grad_cam.grad_cam import GradCAM
from utils.pytorch_grad_cam.utils.image import show_cam_on_image


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default='./test/', help='image to test')
    parser.add_argument('--out-path', type=str, default='./output/', help='mask image to save')
    parser.add_argument('--backbone', type=str, default='mix_transformer',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'mix_transformer'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='./hzh_run/pascal/deeplab-mix_transformer/best_dcn_with_se/model_best.pth.tar',
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
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--cam', type=bool, default=True,
                        help='whether to output Grad-Cam result (default: True)')
    parser.add_argument('--target', type=int, default=1,
                        help='Grad-Cam target result (default: 1)')

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
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    # Grad-Cam
    if args.cam:
        # Please determinate what layer in model you want to hot map
        target_layers = [model.decoder.last_conv]
        cam = GradCAM(model=model, target_layers=target_layers)

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
            # this only for aux(mix_transformer)
            output = output["out"]

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                                3, normalize=False, range=(0, 255))
        save_image(grid_image, args.out_path+"/"+"{}_mask.png".format(name[0:-4]))
        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(name, img_time))

        # Grad-Cam
        if args.cam:
            normalized_masks = torch.softmax(output, dim=1).cpu()
            round_mask = torch.argmax(normalized_masks[0], dim=0).detach().cpu().numpy()
            round_mask_float = np.float32(round_mask == args.target)
            targets = [SemanticSegmentationTarget(args.target, round_mask_float)]

            grayscale_cam = cam(input_tensor=tensor_in, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(np.float32(image) / 255.,
                                              grayscale_cam,
                                              use_rgb=True)
            img = Image.fromarray(visualization)
            img.save(args.out_path + "/" + "{}_mask_cam.png".format(name[0:-4]))

        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in out_path.")


if __name__ == "__main__":
   main()

