import os
from PIL import Image
import numpy as np
import pandas as pd
import torch

from utils import one_hot_encode, logit, csv_to_array

class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, x, labels, logit, dequantize, rng):
            x = self._dequantize(x, rng) if dequantize else x  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x              # logit
            self.labels = labels                                         # numeric labels
            self.y = one_hot_encode(self.labels, 10)                  # 1-hot encoded labels
            self.N = self.x.shape[0]                                       # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x, alpha=1.0e-6):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return logit(alpha + (1 - 2*alpha) * x)                                # number of datapoints


class MnistProcessor():
    def __init__(self, train_path="files\data_files\mnist\mnist_train.csv", 
                 test_path="files\data_files\mnist\mnist_test.csv", 
                 dequantize=False, 
                 logit=False,
                 seed=100):
        self.train_path = train_path
        self.test_path = test_path
        self.dequantize = dequantize
        self.logit = logit
        self.rng = np.random.RandomState(seed)

    def convert_to_tensor(self, data:np.array):
        """将数据转化成张量"""
        return torch.from_numpy(data.astype(np.float32))
    
    def load_data(self, is_train:bool):
        """
        加载数据集
        is_train=True时,加载训练数据,否则加载测试数据
        """
        if is_train:
            data_path = self.train_path
        else:
            data_path = self.test_path
        data = csv_to_array(data_path)
        x = data[:,1:].astype(np.float32) / 255
        labels = data[:,0].astype(np.float32)
        return Data(x, labels, self.logit, self.dequantize, self.rng)

    def onehot_to_num(self, y_onehot):
        """
        将one-hut变量转化成标签
        形状由(n,n_labels)变为(n,)
        """
        y_onehot = np.asarray(y_onehot)

    
        # 如果是一维，先当成单个样本，最后再压平
        if y_onehot.ndim == 1:
            return int(np.argmax(y_onehot))
        
        # 二维情况
        return np.argmax(y_onehot, axis=1).astype(np.int32)
    
def save_one_img(self, x: np.ndarray, 
                 out_dir: str="files\images", 
                 fname: str="复原的图片",
                 logit=False,
                 ):
    """
    将 (1, 784) 的 numpy 数组保存成 28*28 灰度图
    参数
    ----
    img_np : np.ndarray, shape=(1, 784), dtype=float32
        已除以 255 的像素
    out_dir : str
        输出目录
    fname : str
        完整文件名，例如 "test.png"
    """
    assert x.shape == (1, 784), f"shape must be (1, 784), got {x.shape}"
    os.makedirs(out_dir, exist_ok=True)
    # 1. logit → [0,1]   (sigmoid 是 logit 的逆函数)
    if logit:
        x01 = 1 / (1 + np.exp(-x))          # shape 保持 (1, 784)

    x01 = np.clip(x01, 0, 1)

    # 3. 0-1 → 0-255 → uint8
    img = (x01.squeeze() * 255).astype(np.uint8).reshape(28, 28)

    # 4. 保存
    Image.fromarray(img, mode='L').save(os.path.join(out_dir, fname))
    print(f"diffusion 样本已保存 → {os.path.join(out_dir, fname)}")
