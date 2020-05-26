import torch
import torch.nn.functional as F
import numpy as np
from math import log


def nll_loss(output, target):
    return F.nll_loss(output, target)


def quantized_mixture_logistic_loss(x, mixture_params, low=0., high=255., input_channels=3, scaled=True):
    """
    Return
        negative log-likelihood of the input image, which is computed by the learned distribution parameters from the PixelCNN_pp

    Args
        x : input image
        mixture_params : returend parameters for mixture logistic distribution by the PixelCNN_pp
        input_channesls: 3 (for cifar10) => number of mixture should be 10 (i.e last dimension of mixture_params[0])
                         1 (for MNIST) => number of mixture should be 3 (i.e last dimension of mixture_params[0])
        scaled : if True, all values of input images are in [-1, 1]
                 else, [0, 255] => should be re-scaled to be in [-1, 1]
    """
    def log_sum_exp(x):
        """
        Input
            shape [N, H, W, num_mixture]
        Output
            shape [N, H, W]
        """
        max_value, _ = torch.max(x, dim=-1, keepdim=True)

        return max_value.squeeze(-1) + torch.log(torch.sum(torch.exp(x - max_value), dim=-1))

    # log softmax probability from logits
    def log_prob_from_logits(x):
        """
        Input
            shape [N, H, W, num_mixture]
        Output
            shape [N, H, W, num_mixture]
        """
        max_value, _ = torch.max(x, dim=-1, keepdim=True)

        return x - max_value - torch.log(torch.sum(torch.exp(x - max_value), dim=-1, keepdim=True))

    # [N, C, H, W] -> [N, H, W, C] -> [N, H, W, 1, C]
    x = x.permute(0, 2, 3, 1)
    x = x.unsqueeze(-2)
    # scale input value into range [-1, 1]
    if not scaled:
        x = 2. * (x - low) / high - 1.

    cond_1D = (len(mixture_params) == 3 and input_channels == 1)  # input_channel : 1-D, ex) MNIST
    cond_3D = (len(mixture_params) == 4 and input_channels == 3)  # input_channel : 3-D, ex) cifar-10
    assert cond_1D or cond_3D, "# of channel not match with # of params"

    mixture_logit_probs = mixture_params[0]  # shape : [N, H, W, num_mixture]
    means = mixture_params[1]  # shape : [N, H, W, num_mixture, num_channels]
    log_stds = mixture_params[2]  # shape : [N, H, W, num_mixture, num_channels]

    # 3-D input
    if cond_3D:
        coeffs = mixture_params[3]  # shape : [N, H, W, num_mixture, num_coeffs]
        coeffs = F.tanh(coeffs)

        mean_r = means[Ellipsis, 0].unsqueeze(-1)
        mean_g = (means[Ellipsis, 1] + coeffs[Ellipsis, 0] * x[Ellipsis, 0]).unsqueeze(-1)
        mean_b = (means[Ellipsis, 2] + coeffs[Ellipsis, 1] * x[Ellipsis, 0] + coeffs[Ellipsis, 2] * x[
            Ellipsis, 1]).unsqueeze(-1)
        means = torch.cat([mean_r, mean_g, mean_b], dim=-1)  # shape : [N, H, W, num_mixture, num_channels]

    log_stds = torch.clamp(log_stds, min=-7.)
    inverse_stds = torch.exp(-log_stds)
    x_centered = x - means
    # (x + 0.5 - mean) / std
    x_plus = (x_centered + 1 / 255.) * inverse_stds
    # (x - 0.5 - mean) / std
    x_minus = (x_centered - 1 / 255.) * inverse_stds
    x_mid = x_centered * inverse_stds

    """
    * P means cdf function, p means pdf function

    1) low < j < high : P(x=j) = cdf(v <= x_plus) - cdf(v <= x_minus)

    2) j = low : log_P(x=j) = (x - mu + 0.5) /s - log(1 + exp(x - mu + 0.5))
                            = log(1 / (1 + exp(-(x - mu + 0.5)))) 
                            = log_cdf(v <= low) 

    3) j = high : log_P(x=j) = - log(1 + exp(x - mu - 0.5))
                             = log(1 / (1 + exp(x - mu - 0.5)))
                             = log(1 - 1 / (1 + exp(-(x - mu -0.5)))) 
                             = log(1 - cdf(v <= high)) 
                             = log_cdf(P(v >= high))

    4) log_p(x=j) = x - mu -log_s - 2 * log(1 + exp(x - mu))
                  = log(exp(x - mu) / (s * (1 + exp(x - mu))^2)) 
                  = log_pdf(v=mid)
    """
    # all tensors below have shape [N, H, W, num_mixture, num_channels]
    cdf_x_plus = F.sigmoid(x_plus)  # P(v <= x_plus)
    cdf_x_minus = F.sigmoid(x_minus)  # P(v <= x_minus)
    cdf_x_in_range = cdf_x_plus - cdf_x_minus  # P(x) = P(x_minus < v < x_plus), in this case x \in [low, high]
    log_cdf_x_low = x_plus - F.softplus(
        x_plus)  # P(x) = P(v =< x_plus), in this case x = low, so that x_minus -> -inf, as denoted in the paper
    log_cdf_x_high = - F.softplus(
        x_minus)  # P(x) = P(v >= x_minus), in this casd x = high, so that x_plus -> +inf, as denoted in the paper
    log_pdf_x_mid = x_mid - log_stds - 2. * F.softplus(x_mid)

    # due to overflow issue, modify the loss function.
    inner_inner_cond = (cdf_x_in_range > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_x_in_range, min=1e-12)) \
                      + (1. - inner_inner_cond) * (log_pdf_x_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_cdf_x_high + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_x_low + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=-1) + log_prob_from_logits(mixture_logit_probs)

    nll = -torch.sum(log_sum_exp(log_probs))

    return nll

def flow_likelihood_loss(x, pz_det_z_list, input_channels=3):
    log_pz = pz_det_z_list[0]
    log_det = pz_det_z_list[1]

    n_pixel = input_channels * x.shape[2] * x.shape[3]

    nll = -torch.sum(-log(256) * n_pixel + log_det + log_pz)

    return nll
