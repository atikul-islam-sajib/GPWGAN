import torch.nn as nn
import torch.nn.init as init


def weight_initialization(model):
    if isinstance(model, nn.Linear):
        init.normal_(model.weight.data, mean=0, std=0.02)
