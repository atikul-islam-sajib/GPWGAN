import sys
import argparse
import torch

sys.path.append("src/")

from data_loader import Loader
from generator import Generator
from critic import Critic
from trainer import Trainer
from test import Test


def cli():
    """
    # Main CLI for GAN Operations

    This script serves as the main command-line interface (CLI) for various operations related to Generative Adversarial Networks (GANs). It integrates functionalities such as data loading, model training, and synthetic data generation.

    ## Features:
    - Argument parsing for flexible configuration of operations like data loading, training, and synthetic image generation.
    - Facilitates downloading and loading of MNIST dataset.
    - Initiates the training of GAN models.
    - Generates synthetic images using a trained generator.

    ## Usage:
    Run the script from the command line with the desired arguments. For example:
        - `python main_cli.py --download_mnist --batch_size 32 --epochs 100 --latent_space 100 --lr 0.0002 --samples 20`

    Run the script from the command line for synthetic with the desired arguments. For example:
       - `python main_cli.py --samples 20 --latent_space 100 --test`


    ## Arguments:
    - `--batch_size`: Batch size for the DataLoader.
    - `--download_mnist`: Flag to download the MNIST dataset.
    - `--epochs`: Number of epochs for training.
    - `--latent_space`: Dimension of the latent space for the generator.
    - `--lr`: Learning rate for the optimizer.
    - `--samples`: Number of synthetic samples to generate.
    - `--device`: Train the model with CPU, GPU, MPS.
    - `--critic_steps`: Critic steps used to give the priority to the Critic rather Generator.
    - `--display`: Display the critic loss and generator loss in each iterations

    """
    parser = argparse.ArgumentParser(description="Command line coding".title())
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the DataLoader".capitalize(),
    )
    parser.add_argument(
        "--download_mnist",
        action="store_true",
        help="Download Mnist dataset".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--latent_space", type=int, default=100, help="Latent size".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to generate".capitalize(),
    )
    parser.add_argument(
        "--test", action="store_true", help="Run synthetic data tests".capitalize()
    )
    parser.add_argument(
        "--device", default=torch.device("cpu"), help="Device defined".capitalize()
    )
    parser.add_argument(
        "--critic_steps", type=int, default=5, help="Critic steps".capitalize()
    )
    parser.add_argument(
        "--display", default=True, help="Display steps of each training".capitalize()
    )

    args = parser.parse_args()

    if args.download_mnist:
        if (
            args.batch_size > 10
            and args.epochs
            and args.latent_space > 50
            and args.lr
            and args.device
            and args.critic_steps > 1
            and args.display
        ):
            loader = Loader(batch_size=args.batch_size)
            loader.create_loader(mnist_data=loader.download_mnist())

            trainer = Trainer(
                latent_space=args.latent_space,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                n_critic_step=args.critic_steps,
            )
            trainer.train_WGAN()
        else:
            raise Exception("Provide the arguments appropriate way".capitalize())

    if args.test:
        if args.samples % 2 == 0 and args.latent_space > 50:
            test = Test(num_samples=args.samples, latent_space=args.latent_space)
            test.plot_synthetic_image()
        else:
            raise Exception(
                "Please enter a valid number of samples and latent space".capitalize()
            )


if __name__ == "__main__":
    cli()
