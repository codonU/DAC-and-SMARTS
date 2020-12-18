#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *


class BaseNet:
    def __init__(self):
        pass


def layer_init(layer, w_scale=1.0):
    """初始化网络参数

    以正交矩阵初始化权重
    bias 设置为常数0
    """
    # orthogonal 初始化为正交矩阵
    nn.init.orthogonal_(layer.weight.data)      # 要传入data
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
