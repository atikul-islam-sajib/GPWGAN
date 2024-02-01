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
    filename="./logs/critic.log",
)


class Critic(nn.Module):
    """
    A Critic model for evaluating images in a Generative Adversarial Network (GAN).

    This model acts as a critic or discriminator, taking a 2D image and outputting
    a scalar value representing the authenticity of the image. It uses a series of
    linear layers with LeakyReLU activations, except for the output layer which is
    a linear layer. The input images are first flattened before being passed through
    the layers.

    Attributes:
        layers_config (list of tuple): Configuration of the layers where each tuple contains
                                       (in_features, out_features, negative_slope) for LeakyReLU
                                       activated layers, and (in_features, out_features) for the
                                       final output layer.
        model (nn.Sequential): The sequential model comprising the linear and activation layers.

    Raises:
        Exception: If the layers configuration is not provided during the model instantiation,
                   or if the input to the forward pass is None.

    Methods:
        connected_layer(layers_config): Constructs a series of connected layers based on the provided configuration.
        forward(x): Passes the input through the model to evaluate its authenticity.
    """

    def __init__(self):
        """
        Initializes the Critic model and constructs the model layers based on a predefined configuration.
        """
        super(Critic, self).__init__()
        self.layers_config = [
            (28 * 28, 512, 0.2),
            (512, 256, 0.2),
            (256, 128, 0.2),
            (128, 1),
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
                    negative_slope=negative_slope, inplace=True
                )

            (in_features, out_features) = layers_config[-1]
            layers["out_layer"] = nn.Linear(
                in_features=in_features, out_features=out_features
            )

            return nn.Sequential(layers)
        else:
            raise Exception("Layers configuration is not defined in the Critic.")

    def forward(self, x):
        """
        Forward pass of the critic model.

        Args:
            x (Tensor): A batch of 2D images.

        Returns:
            Tensor: A batch of scalar values representing the authenticity of the input images.

        Raises:
            Exception: If the input x is None.
        """
        if x is not None:
            x = x.reshape(-1, 28 * 28)
            x = self.model(x)
        else:
            raise Exception("Input is not defined in the Critic.")
        return x


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
        description="Discriminator script for the MNIST dataset".title()
    )
    parser.add_argument(
        "--critic", action="store_true", help="Discriminator model".capitalize()
    )

    args = parser.parse_args()

    if args.critic:
        discriminator = Critic()
        params = total_params(model=discriminator)
        logging.info(f"Total number of parameters: {params}".capitalize())

    else:
        raise Exception("Discriminator model is not defined".capitalize())
