import torch
import torch.nn as nn
from einops import rearrange

from codebase.model import ConvCoders
from codebase.utils import rotation_utils, model_utils


class RotatingAutoEncoder(nn.Module):
    def __init__(self, opt) -> None:
        """
        Initialize the RotatingAutoEncoder.
        """
        super().__init__()

        self.opt = opt

        # Create model.
        self.encoder = ConvCoders.ConvEncoder(opt)
        self.decoder = ConvCoders.ConvDecoder(opt, self.encoder.channel_per_layer, self.encoder.latent_dim)

        if self.opt.input.dino_processed:
            # Load DINO model and BN preprocess model that goes with it.
            self.dino = model_utils.load_dino_model()
            self.preprocess_model = nn.Sequential(nn.BatchNorm2d(self.opt.input.channel), nn.ReLU())

        # Create output model.
        self.output_weight = nn.Parameter(torch.empty(self.opt.input.channel))
        self.output_bias = nn.Parameter(torch.empty(1, self.opt.input.channel, 1, 1))
        nn.init.constant_(self.output_weight, 1)
        nn.init.constant_(self.output_bias, 0)

    def preprocess_input_images(self, input_images):
        """
        Preprocess input images. Return the images to process (b, n, c, h, w) and
        images to reconstruct (b, c, h, w)
        """
        if self.opt.input.dino_processed:
            with torch.no_grad():
                dino_features = self.dino(input_images)
                dino_features = rearrange(
                    dino_features,
                    "b (h w) c -> b c h w",
                    h=self.opt.input.image_size[0],
                    w=self.opt.input.image_size[1],
                )
            images_to_process = self.preprocess_model(dino_features)
            images_to_reconstruct = dino_features
        else:
            images_to_process = input_images
            images_to_reconstruct = input_images

        if torch.min(images_to_process) < 0:
            raise AssertionError("Error. No negative value allowed in images_to_process.")

        images_to_process = rotation_utils.add_rotation_dimensions(self.opt, images_to_process)
        return images_to_process, images_to_reconstruct

    def encode(self, z):
        """
        Encode the input image from (b, n, c, h, w) to (b, n, c).
        """
        for layer in self.encoder.convolutional:
            z = layer(z)

        z = rearrange(z, "... c h w -> ... (c h w)")
        z = self.encoder.linear(z)
        return z

    def decode(self, z):
        """
        Decode the encoded input image from (b, n, c) to (b, n, c, h, w) and (b, c, h, w).
        (with and without rotation dimensions)
        """
        z = self.decoder.linear(z)

        z = rearrange(
            z,
            "... (c h w) -> ... c h w",
            c=self.encoder.channel_per_layer[-1],
            h=self.encoder.latent_feature_map_size[0],
            w=self.encoder.latent_feature_map_size[1],
        )

        for layer in self.decoder.convolutional:
            z = layer(z)

        reconstruction = self.apply_output_model(torch.linalg.vector_norm(z))

        rotation_output, reconstruction = self.center_crop_reconstruction(z, reconstruction)

        return rotation_output, reconstruction

    def apply_output_model(self, z):
        """
        Apply the output model.
        """
        reconstruction = (torch.einsum("b c h w, c -> b c h w", z, self.output_weight) + self.output_bias)

        if self.opt.input.dino_processed:
            return reconstruction
        else:
            return torch.sigmoid(reconstruction)

    def center_crop_reconstruction(self, rotation_output, reconstruction):
        """
        Do center crop to the reconstructions, in order to match the input image size.
        """
        if self.opt.input.dino_processed:
            rotation_output = rotation_output[:, :, :, 1:-1, 1:-1]
            reconstruction = reconstruction[:, :, 1:-1, 1:-1]
        return rotation_output, reconstruction

    def forward(self, input_images, labels, evaluate):
        """
        Network Forward.
        """
        images_to_process, images_to_reconstruct = self.preprocess_input_images(input_images)

        z = self.encode(images_to_process)
        rotation_output, reconstruction = self.decode(z)

        loss = nn.functional.mse_loss(reconstruction, images_to_reconstruct)

        if evaluate:
            metrics = rotation_utils.run_evaluation(self.opt, rotation_output, labels)
        else:
            metrics = {}

        return loss, metrics
