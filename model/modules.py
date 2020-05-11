import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from functools import partial
from utils import concat_elu
import numpy as np
from utils import *
# dictionary for indicating kernel size, kernel valid regrion 
# "masking convolution" or "normal convolution with shift and crop"
# what is mixture indicator?
# is horizontal stream reference more than 1 row?
# differnece between conditional dependence and corresponding (r, g, b) selection method 
# and mixture logistic and conditioning on the whole image in the pixelCNN++
# where to deploy 3 NIN modules?
# apply dropout

# network in network module
class nin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(nin, self).__init__()
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, 1))
    
    def forward(self, x):
        return self.conv(x)

    
class gated_residual_conv(nn.Module):
    def __init__(self, in_channels, is_horizontal=False, is_decoder=False):
        super(gated_residual_conv, self).__init__()
        self.is_horizontal = is_horizontal
        self.is_decoder = is_decoder
        
        conv_op = down_rightward_conv if is_horizontal else downward_conv
        self.conv_1 = conv_op(2 * in_channels, in_channels)
        self.conv_2 = conv_op(2 * in_channels, 2 * in_channels)
        
        """
        Encoder
            horizontal stream input : previous layer's horizontal output, current layer's vertical output(dim=C)
            vertical stream input : previous layer's vertical output
        
        Decoder 
            horizontal stream input : previous layer's horizontal output, CONCAT(current layer's vertical output(C), 
                                                                                symmetric horizontal output from Encoder)(dim=2C)
            vertical stream input : previous layer's vertical output, symmetric vertical output from Encoder(dim=C)                                         
        """
        
        if self.is_decoder:
            if self.is_horizontal:
                self.nin = nin(2 * 2 * in_channels, in_channels)
            else:
                self.nin = nin(2 * in_channels, in_channels)
        else:
            if self.is_horizontal:
                self.nin = nin(2 * in_channels, in_channels)
        
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x, shortcut_or_vertical_output=None):
        original_x = x
        x = self.conv_1(concat_elu(x))
        
        if shortcut_or_vertical_output is not None:
            x += self.nin(concat_elu(shortcut_or_vertical_output))
        
        x = concat_elu(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x, gate = x.chunk(2, dim=1) # split across the channel dimension 
        x *= torch.sigmoid(gate) # gating x
        
        return original_x + x
        
    
# "down" means "vertical" stream 
# "downright" means "horizontal" stream
class downward_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3), stride=(1, 1), top_pad_output=False):
        super(downward_conv, self).__init__()
        # for vertical, (left, right, top) have to be padded
        self.top_pad_output = top_pad_output
        self.pad = nn.ZeroPad2d((int((kernel_size[1] - 1) / 2), (int((kernel_size[1] - 1) / 2)), kernel_size[0] - 1, 0))
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        
    
        if self.top_pad_output:
            # down shift means removing the last row of output and add padding at the first index of row
            # so that it prevents prediction operation of the last row
            self.down_shift = down_shift
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.down_shift(x) if self.top_pad_output else x
        
        return x
        
    
class down_rightward_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(1, 1), left_pad_output=False):
        super(down_rightward_conv, self).__init__()
        # for horiontal, (left, top) have to be padded
        self.left_pad_output = left_pad_output
        self.pad = nn.ZeroPad2d((kernel_size[1] - 1, 0, kernel_size[0] - 1, 0))
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        
        if self.left_pad_output:
            # right shift means removing the last column of output and add padding at the first index of column
            # so that it prevents prediction operation of the last column
            self.right_shift = right_shift
            
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.right_shift(x) if self.left_pad_output else x
        
        return x
        
        
class downward_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3), stride=(2, 2)):
        super(downward_deconv, self).__init__()
        # output_padding=1 -> add padding to bottom and right
        self.deconv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=1))
        self.kernel_size = kernel_size
    
    def forward(self, x):
        x = self.deconv(x)
        kernel_H, kernel_W = self.kernel_size
        
        # cropping spatial dimension => removing null regions
        return x[Ellipsis, :-(kernel_H - 1), int(np.floor(kernel_W / 2)):-int(np.floor(kernel_W / 2))]

    
class down_rightward_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)):
        super(down_rightward_deconv, self).__init__()
        self.deconv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=1))
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.deconv(x) 
        kernel_H, kernel_W = self.kernel_size
        
        # cropping spatial dimension => removing null regions
        return x[Ellipsis, :-(kernel_H - 1), :-(kernel_W - 1)]

        