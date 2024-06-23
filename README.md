  <h1 align="center">
  Trans-DCN
</h1>
<p align="center">
  A High-Efficiency and Adaptive Deep Network for Bridge Cable Surface Defect Segmentation
</p>

### Introduction
This is a PyTorch(1.8.0) implementation of Trans-DCN. It can use Modified backbone as [train.py](https://github.com/hzh1231/Trans_dcn/blob/master/train.py) mentioned. Currently, we can train Trans-DCN using Pascal VOC 2012, SBD and Cityscapes datasets.

### Installation
The code was tested with Anaconda and Python 3.8. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/hzh1231/Trans_dcn.git
    cd Trans_dcn
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```

### Training
Follow steps below to train your model:

0. Configure your dataset path in [mypath.py](https://github.com/hzh1231/Trans_dcn/blob/master/mypath.py).

1. Input arguments: (see full input arguments via python train.py --help):
  ```Shell
python train.py
 ```

### Testing
Follow steps below to test your model:

0. Configure your test data (images) path in [predict.py](https://github.com/hzh1231/Trans_dcn/blob/master/predict.py).

1. Input arguments: (see full input arguments via python predict.py --help):
  ```Shell
python predict.py
 ```


