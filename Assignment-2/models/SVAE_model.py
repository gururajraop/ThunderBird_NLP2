"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import torch
from torch import nn
import numpy as np
from collections import defaultdict
import time

from .Base_model import BaseModel


class SVAEModel(nn.Module):
    def __init__(self):
        """Initialize the Deterministic RNN Language Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(SVAEModel, self).__init__()

        pass

    def set_input(self, input):
        """load input data from the dataloader.

        Parameters:
            input: includes the input data.
        """
        pass

    def init_hidden(self):
        pass

    def forward(self, input, hidden):
        pass