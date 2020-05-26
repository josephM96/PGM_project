import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from model import *
from math import log, pi


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

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2*pi) - log_sd - 0.5*(x-mean)**2 / torch.exp(2*log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps
