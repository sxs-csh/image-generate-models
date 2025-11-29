
import numpy as np
import pandas as pd
from pathlib import Path

def one_hot_encode(labels, n_labels):
    """
    将类别变量转化成one-hot变量
    原本形状为(n,1)
    转化后形状为(n, n_labels)
    """
    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y

def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))

def csv_to_array(file_path: str,
                 delimiter: str = ',',
                 dtype=float,
                 header: str = 'infer') -> np.ndarray:
    """
    把 CSV 文件读成 np.ndarray
    
    参数
    ----
    file_path : str
        CSV 文件路径
    delimiter : str, optional
        分隔符，默认逗号
    dtype : numpy dtype, optional
        转换后的数据类型，默认 float64
    header : {'infer', None, int}, optional
        是否有表头；'infer' 让 pandas 自动判断，None 表示无表头
    
    返回
    ----
    np.ndarray
    """
    file_path = Path(file_path).expanduser()          # 支持 ~/xxx.csv
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # 用 pandas 读，再转 numpy
    df = pd.read_csv(file_path, delimiter=delimiter, header=header)
    return df.to_numpy(dtype=dtype)