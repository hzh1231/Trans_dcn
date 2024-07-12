import mmcv
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import os

# 这是一个mask图片格式转换程序
# 请依照palette.json在CVAT中配置好类别的RGB颜色
# 将输入的CVAT中的mask导入SegmentationClass_bak文件夹
# 请根据需要输出列别索引，根据palette.json填写在palette中
# 请使用虚拟环境eiesg_env运行程序
# 输出符合VOC格式的mask将保存在SegmentationClass

root = './'
masks = 'SegmentationClass_bak'

palette = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

save_masks = 'SegmentationClass'

if not osp.exists(osp.join(root, save_masks)):
    os.makedirs(osp.join(root, save_masks))

num = 0
for file in mmcv.scandir(osp.join(root, masks), suffix='.png'):
    seg_map = cv2.imread(osp.join(root, masks, file), cv2.IMREAD_GRAYSCALE)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    seg_img.save(osp.join(root, save_masks, file))
    num = num + 1

print("masks转换完毕！数据量为：", num)