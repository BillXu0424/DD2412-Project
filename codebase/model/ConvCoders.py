import torch.nn as nn
from typing import List
from omegaconf import DictConfig

from codebase.model import RotatingLayers
from codebase.utils import model_utils


class ConvEncoder(nn.Module):
    def __init__(self, opt: DictConfig) -> None:
        """
        Initialize the ConvEncoder.
        """
        super().__init__()

        self.channel_per_layer = [
            opt.input.channel,
            opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
        ]

        # Initialize convolutional layers.
        self.convolutional = nn.ModuleList()

        self.convolutional.extend([
            RotatingLayers.RotatingConv2d(
                opt,
                self.channel_per_layer[i // 2],
                self.channel_per_layer[(i + 1) // 2],
                kernel_size=3,
                stride=2 if i % 2 != 0 else 1,
                padding=1,
            )
            for i in range(1, len(self.channel_per_layer) + 1)
        ])

        if opt.input.dino_processed:
            self.convolutional.extend([
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[3],
                    self.channel_per_layer[3],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            ])

        # Initialize linear layer.
        self.latent_feature_map_size, self.latent_dim = model_utils.get_latent_dim(opt, self.channel_per_layer[-1])
        self.linear = RotatingLayers.RotatingLinear(opt, self.latent_dim, opt.model.linear_dim)


class ConvDecoder(nn.Module):
    def __init__(
            self, opt: DictConfig, channel_per_layer: List[int], latent_dim: int,
    ) -> None:
        """
        Initialize the ConvEncoder.
        """
        super().__init__()

        # Initialize convolutional layers.
        self.convolutional = nn.ModuleList()

        for i in range(len(channel_per_layer) + 1, 2, -1):
            if i // 2 == 0:
                self.convolutional.extend([
                    RotatingLayers.RotatingConvTranspose2d(
                        opt,
                        channel_per_layer[(i + 1) // 2],
                        channel_per_layer[i // 2],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                ])
            else:
                self.convolutional.extend([
                    RotatingLayers.RotatingConv2d(
                        opt,
                        channel_per_layer[(i + 1) // 2],
                        channel_per_layer[i // 2],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                ])

        if opt.input.dino_processed:
            self.convolutional.extend([
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[1],
                    channel_per_layer[1],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                )
            ])
            self.convolutional.extend([
                RotatingLayers.RotatingConv2d(
                    opt,
                    channel_per_layer[1],
                    channel_per_layer[0],
                    kernel_size=3,
                    padding=1,
                )
            ])
        else:
            self.convolutional.extend([
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[1],
                    channel_per_layer[0],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                )
            ])

        # Initialize linear layer.
        self.linear = RotatingLayers.RotatingLinear(opt, opt.model.linear_dim, latent_dim)
