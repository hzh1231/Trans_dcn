import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 2
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

'''
what below is Mosaic augmention
'''

import torch.nn.functional as F
import random

def mosaic_segmentation_augmentation(images, masks, num_classes, input_size):
    # images: Tensor of shape (batch_size, num_channels, height, width)
    # masks: Tensor of shape (batch_size, num_classes, height, width)
    # num_classes: Number of segmentation classes
    # input_size: Desired input size of the model

    batch_size, num_channels, height, width = images.shape

    # Initialize an empty mosaic image and mask
    mosaic_image = torch.ones((batch_size, num_channels, input_size[0] * 2, input_size[1] * 2), dtype=images.dtype) * 255
    mosaic_mask = torch.ones((batch_size, num_classes, input_size[0] * 2, input_size[1] * 2), dtype=masks.dtype) * 255

    for i in range(batch_size):
        # Choose 4 images sequentially, and use empty images for the missing ones
        selected_images = images[i:i+4, :, :, :] if i + 4 <= batch_size else torch.ones(4, num_channels, height, width, dtype=images.dtype) * 255
        selected_masks = masks[i:i+4, :, :, :] if i + 4 <= batch_size else torch.ones(4, num_classes, height, width, dtype=masks.dtype) * 255

        # Resize images and masks to half of the input size
        resized_images = F.interpolate(selected_images, scale_factor=0.5, mode='bilinear', align_corners=False)
        resized_masks = F.interpolate(selected_masks, scale_factor=0.5, mode='nearest')

        # Calculate position for each image in the mosaic
        x_offset = (i % 2) * resized_images.shape[3]
        y_offset = (i // 2) * resized_images.shape[2]

        # Paste the resized images and masks onto the mosaic image and mask
        mosaic_image[i:i+1, :, y_offset:y_offset + resized_images.shape[2], x_offset:x_offset + resized_images.shape[3]] = resized_images
        mosaic_mask[i:i+1, :, y_offset:y_offset + resized_masks.shape[2], x_offset:x_offset + resized_masks.shape[3]] = resized_masks

    # Resize the final mosaic image and mask to the desired input size
    mosaic_image = F.interpolate(mosaic_image, size=(input_size[0], input_size[1]), mode='bilinear', align_corners=False)
    mosaic_mask = F.interpolate(mosaic_mask, size=(input_size[0], input_size[1]), mode='nearest')

    return mosaic_image, mosaic_mask

# Example usage:
# images and masks are assumed to be tensors of shape (batch_size, num_channels, height, width)
# num_classes is the number of segmentation classes
# input_size is the desired input size of the model
# mosaic_img, mosaic_mask = mosaic_segmentation_augmentation(images, masks, num_classes, input_size)



