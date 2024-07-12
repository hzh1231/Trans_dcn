import os
import random
import argparse

# 这是一个划分数据集的脚本
# 输入为图像的路径
# 请更改训练、验证、测试集的比例
# 输入为三个集合的txt文件

parser = argparse.ArgumentParser()
# 图像文件的地址，根据自己的数据进行修改图像一般存放在JPEGImages下
parser.add_argument('--images_path', default='JPEGImages/', type=str, help='input images path')
# 数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='ImageSets/Segmentation/', type=str, help='output txt label path')
opt = parser.parse_args()

train_percent = 0.8  # 训练集所占比例
val_percent = 0.1    # 验证集所占比例
test_persent = 0.1   # 测试集所占比例

imagesfilepath = opt.images_path
txtsavepath = opt.txt_path
total_images= os.listdir(imagesfilepath)

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_images)
print("全图像数据量为：", num)
list = list(range(num))     # 存放1-全图像数量

t_train = int(num * train_percent)
print("训练数量为：", t_train)
t_val = int(num * val_percent)
print("测试数量为：", t_val)

train = random.sample(list, t_train)    # 根据train采样数量采样索引
num1 = len(train)       # 理论等于t_train
for i in range(num1):       # 索引去除训练集部分，剩下都是验证集+测试集
    list.remove(train[i])

val_test = [i for i in list if not i in train]
val = random.sample(val_test, t_val)    # 除去训练集后采样验证集
num2 = len(val)
for i in range(num2):
    list.remove(val[i])     # 索引去除训练集部分，剩下都是测试集
print("测试数量为：", len(list))

file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')

for i in train:
    name = os.path.splitext(total_images[i])[0] + '\n'
    file_train.write(name)

for i in val:
    name = os.path.splitext(total_images[i])[0] + '\n'
    file_val.write(name)

for i in list:
    name = os.path.splitext(total_images[i])[0] + '\n'
    file_test.write(name)

file_train.close()
file_val.close()
file_test.close()

print("数据集分割完毕")