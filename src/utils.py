import torch.nn as nn
import torch.nn.init as init


def weight_initialization(model):
    """
    Initializes the weights of a given linear model using a normal distribution.
    This function is typically used to initialize the weights of models in a neural network before training begins.

    Args:
        model (nn.Module): The model whose weights will be initialized. The initialization is performed in-place and only affects linear layers.

    Process:
        - If the model is an instance of nn.Linear, its weights are initialized using a normal distribution with mean 0 and standard deviation 0.02.

    Side Effects:
        - Modifies the model's weights in-place, affecting only the linear layers (nn.Linear instances).

    Example:
        >>> model = nn.Linear(10, 2)
        >>> weight_initialization(model)
        # Model weights are now initialized.
    """
    if isinstance(model, nn.Linear):
        init.normal_(model.weight.data, mean=0, std=0.02)
