import sys
import logging
import argparse
import torch.nn as nn
from collections import OrderedDict

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
    filename="./logs/generator.log",
)


class Generator(nn.Module):
    """
    A Generator model for generating images from a latent space representation.

    This model acts as a generator in a Generative Adversarial Network (GAN),
    taking a random noise vector from a latent space and outputting a 2D image.
    It uses a series of linear layers with LeakyReLU activations, except for the
    output layer which is a linear layer. The final output is reshaped to a 2D image.

    Attributes:
        latent_space (int): The size of the input latent space.
        layers_config (list of tuple): Configuration of the layers where each tuple contains
                                       (in_features, out_features, negative_slope) for LeakyReLU
                                       activated layers, and (in_features, out_features) for the
                                       final output layer.
        model (nn.Sequential): The sequential model comprising the linear and activation layers.

    Args:
        latent_space (int): The size of the latent space from which random inputs are drawn.

    Raises:
        Exception: If the layers configuration is not provided during the model instantiation,
                   or if the input to the forward pass is None.

    Methods:
        connected_layer(layers_config): Constructs a series of connected layers based on the provided configuration.
        forward(x): Passes the input through the model to generate an image.
    """

    def __init__(self, latent_space=100):
        """
        Initializes the Generator model with the given latent space size and constructs
        the model layers based on a predefined configuration.

        Args:
            latent_space (int): The size of the latent space (default: 100).
        """
        self.latent_space = latent_space
        super(Generator, self).__init__()

        self.layers_config = [
            (self.latent_space, 256, 0.2),
            (256, 512, 0.2),
            (512, 1024, 0.2),
            (1024, 28 * 28),
        ]
        self.model = self.connected_layer(self.layers_config)

    def connected_layer(self, layers_config=None):
        """
        Constructs a series of connected layers based on the provided configuration.

        Args:
            layers_config (list of tuple): Layer configurations where each tuple contains
                                           (in_features, out_features, negative_slope) for LeakyReLU
                                           activated layers, and (in_features, out_features) for the
                                           final output layer.

        Returns:
            nn.Sequential: A sequential model comprising the linear and activation layers.

        Raises:
            Exception: If the layers configuration is not provided.
        """
        layers = OrderedDict()

        if layers_config is not None:
            for index, (in_features, out_features, negative_slope) in enumerate(
                layers_config[:-1]
            ):
                layers["{}_layer".format(index)] = nn.Linear(
                    in_features=in_features, out_features=out_features
                )
                layers["{}_activation".format(index)] = nn.LeakyReLU(
                    negative_slope=negative_slope
                )

            (in_features, out_features) = layers_config[-1]
            layers["out_layer"] = nn.Linear(
                in_features=in_features, out_features=out_features
            )

            return nn.Sequential(layers)

        else:
            raise Exception("Layers is not defined in the Geneator".capitalize())

    def forward(self, x):
        """
        Forward pass of the generator model.

        Args:
            x (Tensor): A batch of random noise vectors from the latent space.

        Returns:
            Tensor: A batch of 2D images generated from the input noise vectors.

        Raises:
            Exception: If the input x is None.
        """
        if x is not None:
            x = self.model(x)
        else:
            raise Exception("Input is not defined in the Genearator".capitalize())
        return x.reshape(-1, 1, 28, 28)


if __name__ == "__main__":

    def total_params(model=None):
        """
        Calculates the total number of parameters in a given PyTorch model.

        The function iterates over all parameters in the model and sums their number of elements to get the total parameter count.

        ### Parameters:
        - `model` (torch.nn.Module, optional): The model for which the total number of parameters is to be calculated.

        ### Returns:
        - `total_params` (int): The total number of parameters in the model.

        ### Raises:
        - Exception: If the model is not defined properly (i.e., `model` is None).
        """
        total_params = 0
        if model is not None:
            for _, params in model.named_parameters():
                total_params += params.numel()
        else:
            raise Exception("Model is not defined properly".capitalize())

        return total_params

    parser = argparse.ArgumentParser(
        description="Generator script for the MNIST dataset".title()
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Latent size for dataset".capitalize(),
    )

    parser.add_argument(
        "--generator", action="store_true", help="Generator model".capitalize()
    )

    args = parser.parse_args()

    if args.generator:
        if args.latent_space > 1:
            generator = Generator(latent_space=args.latent_space)
            params = total_params(model=generator)
            logging.info(f"Total number of parameters: {params}".capitalize())
        else:
            raise Exception("Labels size must be greater than 1".capitalize())
    else:
        raise Exception("Generator model is not defined".capitalize())
