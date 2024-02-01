import sys
import logging
import argparse
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.optim as optim

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/trainer.log",
)

from utils import weight_initialization
from generator import Generator
from critic import Critic


class Trainer:
    """
    The `Trainer` class encapsulates the training process for a Wasserstein Generative Adversarial Network (WGAN) composed of a generator and a critic. It manages the training loop, loss computations, parameter updates, and enforces the Lipschitz constraint through weight clipping.

    Attributes:
        latent_space (int): Dimensionality of the generator's input latent space.
        epochs (int): Total number of training epochs.
        learning_rate (float): Learning rate for the Adam optimizers.
        beta1 (float): Beta1 hyperparameter for the Adam optimizer.
        beta2 (float): Beta2 hyperparameter for the Adam optimizer.
        generator (Generator): The generator model of the WGAN.
        critic (Critic): The critic model of the WGAN.
        dataloader (DataLoader): DataLoader providing the training data.
        optimizer_generator (optim.Optimizer): Optimizer for updating the generator's weights.
        optimizer_critic (optim.Optimizer): Optimizer for updating the critic's weights.
        critic_loss (list): List to record the critic's loss after each epoch.
        generator_loss (list): List to record the generator's loss after each epoch.
        device (str): The device ('cuda', 'mps', or 'cpu') on which the models will run.
        n_critic_step (int): Number of critic updates per generator update.
        clamp_value (float): The clamp value for the critic's weight clipping.

    Methods:
        connect_gpu(generator, critic, device): Assigns the generator and critic models to the specified device.
        saved_checkpoints(model, epoch): Saves a checkpoint of the model at the specified epoch.
        train_critic(real_samples, fake_samples): Trains the critic model for one batch of data.
        train_generator(generated_samples): Trains the generator model for one batch of data.
        train_WGAN(): Conducts the training loop for the WGAN.
    """

    def __init__(
        self,
        latent_space=100,
        epochs=100,
        lr=0.00005,
        beta1=0.5,
        beta2=0.999,
        device="cpu",
        n_critic_step=5,
    ):
        """
        Initializes the Trainer object with the specified configuration and sets up the neural network models, dataloader, loss function, and optimizers.

        Args:
            latent_space (int): Size of the latent space (input vector for the generator).
            epochs (int): Number of epochs for training the models.
            lr (float): Learning rate for the Adam optimizers.
            beta1 (float): Beta1 hyperparameter for the Adam optimizer.
            beta2 (float): Beta2 hyperparameter for the Adam optimizer.
            device (str): The device ('cuda', 'mps', or 'cpu') on which the models will run.
            n_critic_step (int): Number of critic updates per generator update.
        """
        self.latent_space = latent_space
        self.epochs = epochs
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device
        self.clamp_value = 0.01
        self.n_critic_step = n_critic_step

        self.device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_mps = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device_cpu = torch.device("cpu")

        self.generator = Generator()
        self.critic = Critic()

        self.generator, self.critic, self.device = self.connect_gpu(
            generator=self.generator, critic=self.critic, device=self.device
        )

        self.generator.apply(weight_initialization)
        self.critic.apply(weight_initialization)

        try:
            self.dataloader = joblib.load(filename="./data/processed/dataloader.pkl")
        except Exception as e:
            logging.exception("Dataloader is not transformed from pickle".capitalize())

        self.optimizer_generator = optim.Adam(
            params=self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        self.optimizer_critic = optim.Adam(
            params=self.critic.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        self.critic_loss = list()
        self.generator_loss = list()

    def connect_gpu(self, generator, critic, device):
        """
        Connects the generator and critic models to the specified computing device.

        Args:
            generator (Generator): The generator model.
            critic (Critic): The critic model.
            device (str): The target device ('cuda', 'mps', or 'cpu').

        Returns:
            tuple: The generator, critic, and device after assignment to the target device.
        """
        if device == "cuda":
            generator = generator.to(self.device_cuda)
            critic = critic.to(self.device_cuda)
        elif device == "mps":
            generator = generator.to(self.device_mps)
            critic = critic.to(self.device_mps)
        else:  # Default to CPU if neither 'cuda' nor 'mps'
            generator = generator.to(self.device_cpu)
            critic = critic.to(self.device_cpu)

        return generator, critic, device

    def saved_checkpoints(self, model=None, epoch=None):
        """
        Saves a checkpoint of the given model at the specified epoch.

        Args:
            model (nn.Module): The model to be saved.
            epoch (int): The current epoch number for naming the saved file.

        Side Effects:
            Saves the model's state dictionary to the file system.
        """
        try:
            torch.save(
                model.state_dict(), f"./models/checkpoints/generator_{epoch}.pth"
            )
            logging.info(f"Checkpoint saved for epoch {epoch}")
        except Exception as e:
            logging.error(f"Error saving checkpoint at epoch {epoch}: {e}")
            raise e

    def train_critic(self, real_samples, fake_samples):
        """
        Trains the critic model for one batch of data.

        Args:
            real_samples (Tensor): Real samples from the dataset.
            fake_samples (Tensor): Fake samples generated by the generator.

        Returns:
            float: The total loss for the critic for the current batch.

        Side Effects:
            Updates the weights of the critic model.
        """
        real_predict = self.critic(real_samples)
        fake_predict = self.critic(fake_samples.detach())

        total_loss = -torch.mean(real_predict) + torch.mean(fake_predict)

        self.optimizer_critic.zero_grad()
        total_loss.backward()
        self.optimizer_critic.step()

        return total_loss.item()

    def train_generator(self, generated_samples):
        """
        Trains the generator model for one batch of data.

        Args:
            generated_samples (Tensor): Samples generated by the generator.

        Returns:
            float: The loss for the generator for the current batch.

        Side Effects:
            Updates the weights of the generator model.
        """
        generated_loss = -torch.mean(self.critic(generated_samples))

        self.optimizer_generator.zero_grad()
        generated_loss.backward()
        self.optimizer_generator.step()

        return generated_loss.item()

    def train_WGAN(self):
        """
        Conducts the training loop for the Wasserstein Generative Adversarial Network (WGAN).
        The loop iterates over the dataset, trains the critic and generator in alternation,
        and records the loss for each epoch.

        Process:
        - For each epoch:
            - For each batch in the dataloader:
                - Train the critic using both real and fake data.
                - Generate new fake samples and train the generator.
                - Record and accumulate the loss for both the critic and generator.
        - After each epoch, print the average losses and save the generator's state as a checkpoint.

        Side Effects:
        - Updates the weights of both the critic and generator models.
        - Appends the average loss of each epoch to the respective loss lists (`critic_loss`, `generator_loss`).
        - Saves the generator's state after each epoch.
        - Prints the progress and average losses to the console.

        Error Handling:
        - If the model checkpoint cannot be saved, an exception is raised.
        """
        for epoch in range(self.epochs):
            c_loss = []
            g_loss = []
            for index, (real_samples, _) in enumerate(self.dataloader):
                real_samples = real_samples.to(self.device)
                batch_size = real_samples.shape[0]

                noise_samples = torch.randn(batch_size, self.latent_space).to(
                    self.device
                )
                fake_samples = self.generator(noise_samples)

                D_loss = self.train_critic(
                    real_samples=real_samples, fake_samples=fake_samples
                )
                c_loss.append(D_loss)

                # Clamp the weights of the critic
                for params in self.critic.parameters():
                    params.data.clamp_(-self.clamp_value, self.clamp_value)

                if (index + 1) % self.n_critic_step == 0:
                    generated_samples = self.generator(noise_samples)
                    G_loss = self.train_generator(generated_samples=generated_samples)
                    g_loss.append(G_loss)

            avg_critic_loss = np.mean(c_loss)
            avg_generator_loss = np.mean(g_loss)
            self.critic_loss.append(avg_critic_loss)
            self.generator_loss.append(avg_generator_loss)

            logging.info(f"Epoch [{epoch + 1}/{self.epochs}] Completed")
            logging.info(
                f"Average Critic Loss: {avg_critic_loss:.4f}, Average Generator Loss: {avg_generator_loss:.4f}"
            )

            self.saved_checkpoints(model=self.generator, epoch=epoch + 1)


if __name__ == "__main__":
    """
    # GAN Trainer Script

    This script is responsible for setting up and running the training process of a Generative Adversarial Network (GAN). It includes argument parsing for command-line customization of the training parameters and initiates the training loop for the GAN models.

    ## Features:
    - Command-line argument parsing for flexible training configuration.
    - Conditional checks to ensure valid training parameters.
    - Logging for monitoring the training process.
    - Integration with the `Trainer` class to facilitate the actual training.

    ## Usage:
    To use this script, run it from the command line with the desired arguments, for example:
        python trainer_script.py --epochs 100 --latent_space 100 --lr 0.0002
    ## Arguments:
    - `--epochs`: Number of epochs for training.
    - `--latent_space`: Size of the latent space for the generator.
    - `--lr`: Learning rate for the optimizer.
    """

    parser = argparse.ArgumentParser(description="GAN Training".title())
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
        "--device", default=torch.device("cpu"), help="Device defined".capitalize()
    )
    parser.add_argument(
        "--critic_steps", type=int, default=5, help="Critic steps".capitalize()
    )

    args = parser.parse_args()

    if (
        args.epochs > 1
        and args.latent_space > 50
        and args.lr
        and args.device
        and args.critic_steps
    ):
        logging.info("Training started".capitalize())

        trainer = Trainer(
            latent_space=args.latent_space,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            n_critic_step=args.critic_steps,
        )
        trainer.train_WGAN()

        logging.info("Training completed successfully".capitalize())
    else:
        raise Exception("Please check the arguments".capitalize())
