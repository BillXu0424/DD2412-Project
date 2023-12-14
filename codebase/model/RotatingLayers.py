import math

import torch
import torch.nn as nn

from codebase.utils import rotation_utils

"""
看注释
"""


def apply_layer_to_rotating_features(opt, layer, rotation_bias, norm, x):
    """
    Apply layer to rotating features.
    """
    psi = None
    if isinstance(layer, torch.nn.Linear):
        psi = layer(x)  # (b, n, c).
    elif isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        n = opt.model.rotation_dimensions
        b = opt.input.batch_size

        x_folded = torch.flatten(x, start_dim=0, end_dim=1)
        psi_folded = layer(x_folded)

        psi = torch.reshape(psi_folded, (b, n, psi_folded.shape[1], psi_folded.shape[2], psi_folded.shape[3]))
    else:
        print("Error. Layer is not a valid instance.")

    z = psi + rotation_bias

    magnitude_psi = torch.linalg.vector_norm(psi)
    chi = layer(torch.linalg.vector_norm(x))
    magnitude = 0.5 * (magnitude_psi + chi)
    magnitude = nn.functional.relu(norm(magnitude))

    res = rotation_utils.rescale_magnitude_rotating_features(z, magnitude)
    return res


def init_rotation_bias(fan_in, bias):
    """
    Init the rotation bias.
    """
    bound = 1 / math.sqrt(fan_in)
    return torch.nn.init.uniform_(bias, -bound, bound)


class RotatingConvTranspose2d(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        """
        Init a 2D transposed rotating features conv layer.
        """
        super().__init__()

        self.opt = opt
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                             bias=False)
        self.norm = nn.BatchNorm2d(out_channels, affine=True)
        self.fan_in = (out_channels * self.trans_conv.kernel_size[0] * self.trans_conv.kernel_size[1])
        self.r_b = nn.Parameter(torch.empty((1, opt.model.rotation_dimensions, out_channels, 1, 1)))
        self.r_b = init_rotation_bias(self.fan_in, self.r_b)

    def forward(self, x):
        """
        Forward pass.
        """
        return apply_layer_to_rotating_features(self.opt, self.trans_conv, self.r_b, self.norm, x)


class RotatingConv2d(nn.Module):
    def __init__(self, opt, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
           Initi a 2D rotating features conv layer.
        """
        super().__init__()

        self.opt = opt
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, affine=True)
        self.fan_in = in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]
        self.r_b = nn.Parameter(torch.empty((1, opt.model.rotation_dimensions, out_channels, 1, 1)))
        self.r_b = init_rotation_bias(self.fan_in, self.r_b)

    def forward(self, x):
        """
        Forward pass.
        """
        return apply_layer_to_rotating_features(self.opt, self.conv, self.r_b, self.norm, x)


class RotatingLinear(nn.Module):
    def __init__(self, opt, in_features, out_features):
        """
        Initialize a rotating features linear layer.
        """
        super().__init__()

        self.opt = opt
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features, elementwise_affine=True)
        self.fan_in = in_features
        self.r_b = nn.Parameter(torch.empty((1, opt.model.rotation_dimensions, out_features)))
        self.r_b = init_rotation_bias(self.fan_in, self.r_b)

    def forward(self, x):
        """
        Forward pass.
        """
        return apply_layer_to_rotating_features(self.opt, self.fc, self.r_b, self.norm, x)
