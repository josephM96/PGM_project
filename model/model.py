import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, TransformedDistribution, Independent, Uniform
from torch.distributions import SigmoidTransform, AffineTransform, Transform
import torchvision
from modules import *
from utils import *
# quantized distribution 어떻게 할 것인지


class PixelCNN_EncoderLayer(nn.Module):
    def __init__(self, num_gated_residual_conv=5, num_filters=80):
        super(PixelCNN_EncoderLayer, self).__init__()
        self.num_gated_residual_conv = num_gated_residual_conv
        
        # vertical stream
        self.downward_stream = nn.ModuleList([gated_residual_conv(num_filters, 
                                                                  is_horizontal=False, 
                                                                  is_decoder=False)
                                             for _ in range(self.num_gated_residual_conv)])
        # horizontal stream
        self.down_rightward_stream = nn.ModuleList([gated_residual_conv(num_filters, 
                                                                        is_horizontal=True, 
                                                                        is_decoder=False)
                                                   for _ in range(self.num_gated_residual_conv)])
    def forward(self, downward_input, down_rightward_input):
        downward_cache = []
        down_rightward_cache = []
        
        # caching the output of the each stream in a block(e.g. in 32x32 block)
        for i in range(self.num_gated_residual_conv):
            downward = self.downward_stream[i](downward_input)
            down_rightward = self.down_rightward_stream[i](down_rightward_input,
                                                           shortcut_or_vertical_output=downward)
            downward_cache.append(downward)
            down_rightward_cache.append(down_rightward)
        
        return downward_cache, down_rightward_cache
        

class PixelCNN_DecoderLayer(nn.Module):
    def __init__(self, num_gated_residual_conv=5, num_filters=80):
        super(PixelCNN_DecoderLayer, self).__init__()
        self.num_gated_residual_conv = num_gated_residual_conv
        
        # vertical stream
        self.downward_stream = nn.ModuleList([gated_residual_conv(num_filters, 
                                                                  is_horizontal=False, 
                                                                  is_decoder=True)
                                             for _ in range(self.num_gated_residual_conv)])
        
        # horizontal stream
        self.down_rightward_stream = nn.ModuleList([gated_residual_conv(num_filters, 
                                                                        is_horizontal=True, 
                                                                        is_decoder=True)
                                                   for _ in range(self.num_gated_residual_conv)])
    
    def forward(self, downward_input, down_rightward_input, downward_cache, down_rightward_cache):
        for i in range(self.num_gated_residual_conv):
            downward = self.downward_stream[i](downward_input, 
                                               shortcut_or_vertical_output=downward_cache.pop())

            down_rightward = self.down_rightward_stream[i](down_rightward_input, 
                                                           shortcut_or_vertical_output=
                                                           torch.cat([downward, down_rightward_cache.pop()], dim=1))

        return downward, down_rightward


class PixelCNN_pp(nn.Module):
    def __init__(self, num_gated_residual_conv=5, num_filters=80, num_scales=3, num_mixture=10, input_channels=3):
        super(PixelCNN_pp, self).__init__()
        self.num_filters = num_filters
        self.num_scales = num_scales
        self.num_mixture = num_mixture
        self.input_channels = input_channels
        num_residuals = [num_gated_residual_conv, num_gated_residual_conv + 1, num_gated_residual_conv + 1]
        
        self.Encoder = nn.ModuleList([PixelCNN_EncoderLayer(num_gated_residual_conv, self.num_filters)
                                     for _ in range(self.num_scales)])
        self.Decoder = nn.ModuleList([PixelCNN_DecoderLayer(num_residuals[i], self.num_filters)
                                     for i in range(self.num_scales)])
        
        # down/up sizing for hierarchical resolution
        self.downsize_downward = nn.ModuleList([downward_conv(num_filters, num_filters, stride=(2, 2))
                                               for _ in range(self.num_scales - 1)])
        self.downsize_down_rightward = nn.ModuleList([down_rightward_conv(num_filters, num_filters, stride=(2, 2))
                                                     for _ in range(self.num_scales - 1)])
        self.upsize_downward = nn.ModuleList([downward_deconv(num_filters, num_filters, stride=(2, 2))
                                             for _ in range(self.num_scales -1)])
        self.upsize_down_rightward = nn.ModuleList([down_rightward_deconv(num_filters, num_filters, stride=(2, 2))
                                                   for _ in range(self.num_scales - 1)])
        
        # initial stream for vertical and horizontal
        # for vertical
        # we remove the last row of the conv output and add the padding to the first row(i.e. down_shift) -> instead of masking!
        # for horizontal
        # we remove the last columns of the conv output and add the padding to the first column(i.e. right_shift) -> instead of masking!
        # but why add padding to channel??
        # Also why horizontal init with vertical_kernel(1, 3) and horizontal_init(2, 1)???
        self.downward_init = downward_conv(input_channels + 1, num_filters, kernel_size=(2, 3), top_pad_output=True)
        self.down_rightward_init = nn.ModuleList([downward_conv(input_channels + 1, 
                                                                num_filters, 
                                                                kernel_size=(1, 3), 
                                                                top_pad_output=True),
                                                  down_rightward_conv(input_channels + 1, 
                                                                      num_filters, 
                                                                      kernel_size=(2, 1), 
                                                                      left_pad_output=True)])
        
        self.num_coeffs = self.input_channels * (self.input_channels - 1) // 2 # nC2  /2
        self.num_out = self.input_channels * 2 + self.num_coeffs + 1 # mu, s, c_rg, c_rb, c_gb, pi
        
        self.output_layer = nin(num_filters, self.num_mixture * self.num_out)
        
    def forward(self, x):
        x_shape = [int(dim) for dim in x.size()]
        init_padding = torch.ones(x_shape[0], 1, x_shape[2], x_shape[3])
        init_padding = init_padding.cuda() if x.is_cuda else init_padding
        
        x = torch.cat([x, init_padding], dim=1)
        
        # Encoding Path
        downward_cache = [self.downward_init(x)]
        down_rightward_cache = [self.down_rightward_init[0](x) + self.down_rightward_init[1](x)]
        for i in range(self.num_scales):
            downward_list, down_rightward_list = self.Encoder[i](downward_cache[-1], down_rightward_cache[-1])
            downward_cache += downward_list
            down_rightward_cache += down_rightward_list
            
            # At the last block we should not downscale the tensor
            if i != 2:
                downward = self.downsize_downward[i](downward_cache[-1])
                down_rightward = self.downsize_down_rightward[i](down_rightward_cache[-1])
                downward_cache.append(downward)
                down_rightward_cache.append(down_rightward)

        # Decoding Path
        downward = downward_cache.pop()
        down_rightward = down_rightward_cache.pop()
        for i in range(self.num_scales):
            downward, down_rightward = self.Decoder[i](downward, down_rightward, downward_cache, down_rightward_cache)
            
            # At the last block we should not upscale the tensor
            if i != 2:
                downward = self.upsize_downward[i](downward)
                down_rightward = self.upsize_down_rightward[i](down_rightward)
        
        mixture_params = self.output_layer(F.elu(down_rightward))
        # mixture params shape : [N, C, H, W, num_out * num_mixture]
        mixture_params = mixture_params.permute(0, 2, 3, 1).reshape(-1, x_shape[2], x_shape[3], self.num_mixture, self.num_out)
        
        split = [1, 1, 1] if self.input_channels == 1 else [1, self.input_channels, self.input_channels, self.num_coeffs]
        mixture_params = list(torch.split(mixture_params, split, dim=-1))
        mixture_params[0] = torch.squeeze(mixture_params[0], dim=-1)
        
        return mixture_params
