# GPWGAN - Generative Adversarial Network for MNIST Dataset: Wasserstein Generative Models

## Overview

GPWGAN (GAN-based Project for Synthesizing Grayscale images) is a machine learning project focused on generating synthetic images using Generative Adversarial Networks (GANs). Specifically, it is designed to work with the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems.

## Features

- Utilizes PyTorch for implementing GAN models.
- Provides scripts for easy training and generating synthetic images.
- Command Line Interface for easy interaction.
- Includes a custom data loader for the MNIST dataset.
- Customizable training parameters for experimenting with GAN.

## Installation

Clone the repository:

```
git clone https://github.com/atikul-islam-sajib/GPWGAN.git

cd GPWGAN
```

# Install dependencies

```
pip install -r requirements.txt
```

## Usage

Examples of commands and their explanations.

```bash
python /path/to/GPWGAN/src/cli.py --help
```

### Options

- `--batch_size BATCH_SIZE`: Set the batch size for the dataloader. (Default: specify if there's one)
- `--download_mnist`: Download the MNIST dataset.
- `--epochs EPOCHS`: Set the number of training epochs.
- `--latent_space LATENT_SPACE`: Define the size of the latent space for the model.
- `--lr LR`: Specify the learning rate for training the model.
- `--samples SAMPLES`: Determine the number of samples to generate after training.
- `--test`: Run tests with synthetic data to validate model performance.
- `--device`: Train the model with CPU, GPU, MPS.
- `--critic_steps`: Critic steps used to give the priority to the Critic rather Generator.
- `--display`: Display the critic loss and generator loss in each iterations[True/False]

## Training and Generating Images(CLI)

### Training the GAN Model

To train the GAN model with default parameters with mps:

```
!python /content/GPWGAN/src/cli.py  --epochs 200 --latent_space 100 --lr 0.00005 --device mps --batch_size 64 --critic_steps 5 --download_mnist --display True
```

To train the GAN model with default parameters with gpu:

```
!python /content/GPWGAN/src/cli.py  --epochs 200 --latent_space 100 --lr 0.00005 --device gpu --batch_size 64 --critic_steps 5 --download_mnist --display True
```

To train the GAN model with default parameters with cpu:

```
!python /content/GPWGAN/src/cli.py  --epochs 200 --latent_space 100 --lr 0.00005 --device cpu --batch_size 64 --critic_steps 5 --download_mnist --display True
```

### Generating Images

To generate images using the trained model:

```
!python /content/GPWGAN/src/cli.py --samples 20 --latent_space 100 --test
```

### Viewing Generated Images

Check the specified output directory for the generated images.

```
from IPython.display import Image
Image(filename='/content/GPWGAN/outputs/generated_image.png')
```

## Core Script Usage

The core script sets up the necessary components for training the GAN. Here's a quick overview of what each part does:

```python
from src.data_loader import Loader
from src.train import Trainer
from src.test import Test

# Initialize the data loader with batch size
loader = Loader(batch_size = 64)
loader.create_loader(mnist_data = loader.download_mnist())

# Set up the trainer with learning rate, epochs, and latent space size
trainer = Trainer(latent_space = 100, epochs = 10, lr = 0.0005, device = 'gpu', batch_size = 64, n_critic_step = 5, clip_value = 0.01)
trainer.train_WGAN()

# Test the generated dataset and display the synthetic images
test = Test(num_samples = 20,latent_space = 100)
test.plot_synthetic_image()
```

This script initializes the data loader, downloads the MNIST dataset, and prepares the data loader. It then sets up and starts the training process for the GAN model.

## Documentation

For detailed documentation on the implementation and usage, visit the [GPCGAN Documentation](https://atikul-islam-sajib.github.io/GPCGAN-deploy/).

## Notebook Training

For detailed documentation on the implementation and usage using notebook, visit the [Notebook](./notebooks/ModelTrain-CGAN.ipynb).

## Contributing

Contributions to improve the project are welcome. Please follow the standard procedures for contributing to open-source projects.

## License

This project is licensed under [MIT LICENSE](./LICENSE). Please see the LICENSE file for more details.

## Acknowledgements

Thanks to all contributors and users of the GPWGAN project. Special thanks to those who have provided feedback and suggestions for improvements.

## Contact

For any inquiries or suggestions, feel free to reach out to [atikulislamsajib137@gmail.com].

## Additional Information

- This project is a work in progress and subject to changes.
- Feedback and suggestions are highly appreciated.
