import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from model import *


def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))


def down_shift(x):
    x = x[Ellipsis, :-1, :]
    x = nn.ZeroPad2d((0, 0, 1, 0))(x)

    return x


def right_shift(x):
    x = x[Ellipsis, :-1]
    x = nn.ZeroPad2d((1, 0, 0, 0))(x)

    return x