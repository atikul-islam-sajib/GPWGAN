import logging
import argparse
import sys
import joblib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filemode="w",
    filename="./logs/data_loader.log",
)


class Loader:
    """
    The `Loader` class is responsible for downloading the MNIST dataset and preparing a DataLoader for it.

    The class initializes with a batch size and provides methods to download the MNIST dataset and create a DataLoader object. The DataLoader can then be used to iterate over the dataset in batches.

    ## Attributes:
    - `batch_size` (int): The size of the batches to divide the dataset into.

    ## Methods:
    - `__init__(self, batch_size=32)`: Constructor for the Loader class.
    - `download_mnist(self)`: Downloads the MNIST dataset and applies transformations.
    - `create_loader(self, mnist_data=None)`: Creates a DataLoader from the MNIST dataset.
    """

    def __init__(self, batch_size=32):
        """
        Initializes the Loader class with a specified batch size.

        ### Parameters:
        - `batch_size` (int): The size of the batches in which the dataset will be split. Default is 32.
        """
        self.batch_size = batch_size

    def download_mnist(self):
        """
        Downloads the MNIST dataset and applies necessary transformations.

        The method applies a composition of transformations to the dataset: converting images to tensors and normalizing them.

        ### Returns:
        - `mnist_data` (Dataset): The MNIST dataset with applied transformations.
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        mnist_data = datasets.MNIST(
            root="./data/raw/",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        return mnist_data

    def create_loader(self, mnist_data=None):
        """
        Creates a DataLoader from the provided MNIST dataset.

        The method initializes a DataLoader with the given dataset and batch size. It also handles the saving of the DataLoader object using joblib. Any exceptions during DataLoader creation or saving are logged.

        ### Parameters:
        - `mnist_data` (Dataset, optional): The MNIST dataset to be loaded into the DataLoader. If not provided, an exception is logged.

        ### Side Effects:
        - Creates a DataLoader object and saves it to a file.
        - Logs exceptions if the DataLoader creation or saving fails.
        """
        if mnist_data is not None:
            dataloader = DataLoader(
                mnist_data, batch_size=self.batch_size, shuffle=True
            )
            try:
                logging.info("Saving dataloader".capitalize())

                joblib.dump(
                    value=dataloader, filename="./data/processed/dataloader.pkl"
                )
            except Exception as e:
                logging.exception(
                    "Error occurred while creating dataloader".capitalize()
                )
        else:
            logging.exception("No data provided".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and save a DataLoader from MNIST dataset".title()
    )
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

    args = parser.parse_args()

    if args.download_mnist:
        if args.batch_size > 10:
            loader = Loader(batch_size=args.batch_size)
            loader.create_loader(mnist_data=loader.download_mnist())
        else:
            logging.warning(
                "Batch size is less than 10. Consider using a larger batch size for better performance.".capitalize()
            )
    else:
        logging.exception(
            "MNIST dataset not found. Please download the dataset.".capitalize()
        )
