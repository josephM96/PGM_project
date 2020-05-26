import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from functools import partial
from model.model_utils import concat_elu
import numpy as np
from model.model_utils import *

from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))

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
        x, gate = x.chunk(2, dim=1)  # split across the channel dimension
        x *= torch.sigmoid(gate)  # gating x

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
        self.deconv = weight_norm(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=1))
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.deconv(x)
        kernel_H, kernel_W = self.kernel_size

        # cropping spatial dimension => removing null regions
        return x[Ellipsis, :-(kernel_H - 1), int(np.floor(kernel_W / 2)):-int(np.floor(kernel_W / 2))]


class down_rightward_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)):
        super(down_rightward_deconv, self).__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=1))
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.deconv(x)
        kernel_H, kernel_W = self.kernel_size

        # cropping spatial dimension => removing null regions
        return x[Ellipsis, :-(kernel_H - 1), :-(kernel_W - 1)]


#################### From this line, modules for Glow ####################

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.shift = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))
            std = (flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))

            self.shift.data.copy_(-mean)
            self.scale.data.copy_(1/(std+1e-6))

    def forward(self, input):
        """ x(data) -> z(latent) """
        _, _, H, W = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = H * W * torch.sum(log_abs)

        if self.logdet:
            return self.scale*(input+self.shift), logdet
        else:
            return self.scale * (input + self.shift)

    def reverse(self, output):
        """ z(latent) -> x(data) """
        return output / self.scale - self.shift


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, H, W = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            H * W * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)  # rotation matrix
        w_l = torch.from_numpy(w_l)  # lower triangular matrix with 1 as diagonal entries
        w_s = torch.from_numpy(w_s)  # diagonal entries of w_u
        w_u = torch.from_numpy(w_u)  # zero diagonal entries of w_u(upper triangular matrix)

        self.register_buffer('w_p', w_p)  # rotation matrix, fixed
        self.register_buffer('u_mask', torch.from_numpy(u_mask)) # upper triangular matrix as mask 1 without diagonal entries.
        self.register_buffer('l_mask', torch.from_numpy(l_mask)) # lower triangular matrix as mask 1 without diagonal entries.
        self.register_buffer('s_sign', torch.sign(w_s))  # sign of diagonal entries of s
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))  # identity matrix as size c x c
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, H, W = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = H*W*torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight =(self.w_p
                 @(self.w_l*self.l_mask + self.l_eye)
                 @((self.w_u*self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))))

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale*3)

        return out


class CouplingLayer(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel//2, filter_size, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel//2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s + 2)
            s = torch.sigmoid(log_s + 2)
            # out_b = s * in_b + t
            out_b = (in_b + t) * s

            # logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], dim=1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s + 2)
            s = torch.sigmoid(log_s + 2)
            # in_b = (out_b - t)/s
            in_b = out_b/s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], dim=1)

class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = CouplingLayer(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, logdet1 = self.invconv(out)
        out, logdet2 = self.coupling(out)

        logdet = logdet + logdet1
        if logdet2 is not None:
            logdet = logdet + logdet2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input

class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, learned_prior=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList([])
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split
        self.learned_prior = learned_prior

        if learned_prior:
            if split:
                self.prior = ZeroConv2d(in_channel*2, in_channel*4)
            else:
                self.prior = ZeroConv2d(in_channel*4, in_channel*8)
        else:
            if split:
                self.prior = torch.zeros((1, in_channel*4, 1, 1))
            else:
                self.prior = torch.zeros((1, in_channel*8, 1, 1))

    def forward(self, input):
        B, C, H, W = input.size()
        out = self.squeeze(input)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            if self.learned_prior:
                mean, log_sd = self.prior(out).chunk(2, 1)
            else:
                mean, log_sd = self.prior.chunk(2, 1)
                mean = mean.repeat(B, 1, *out.shape[2:]).to(out.device)
                log_sd = log_sd.repeat(B, 1, *out.shape[2:]).to(out.device)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(B, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            if self.learned_prior:
                mean, log_sd = self.prior(zero).chunk(2, 1)
            else:
                mean, log_sd = self.prior.chunk(2, 1)
                mean = mean.repeat(B, 1, *out.shape[2:]).to(out.device)
                log_sd = log_sd.repeat(B, 1, *out.shape[2:]).to(out.device)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(B, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        """ z(latent) -> x(data) """
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                if self.learned_prior:
                    mean, log_sd = self.prior(input).chunk(2, 1)
                else:
                    mean, log_sd = self.prior.chunk(2, 1)
                    mean = mean.repeat(eps.shape[0], 1, *eps.shape[2:]).to(eps.device)
                    log_sd = log_sd.repeat(eps.shape[0], 1, *eps.shape[2:]).to(eps.device)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                if self.learned_prior:
                    mean, log_sd = self.prior(zero).chunk(2, 1)
                else:
                    mean, log_sd = self.prior.chunk(2, 1)
                    mean = mean.repeat(eps.shape[0], 1, *eps.shape[2:]).to(eps.device)
                    log_sd = log_sd.repeat(eps.shape[0], 1, *eps.shape[2:]).to(eps.device)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

            for flow in self.flows[::-1]:
                input = flow.reverse(input)

        input = self.undo_squeeze(input)
        return input

    def squeeze(self, input):
        B, C, H, W = input.size()
        squeezed = input.view(B, C, H//2, 2, W//2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view((B, C*4, H//2, W//2))
        return out

    def undo_squeeze(self, output):
        B, C, H, W = output.size()
        unsqueezed = output.view(B, C//4, 2, 2, H, W)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        out = unsqueezed.contiguous().view((B, C//4, H*2, W*2))
        return out
