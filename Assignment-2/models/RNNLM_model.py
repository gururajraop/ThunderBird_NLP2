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


class RNNLMModel(nn.Module):
    def __init__(self, opt, vocab_size):
        """Initialize the Deterministic RNN Language Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(RNNLMModel, self).__init__()

        self.opt = opt
        self.vocab_size = vocab_size
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.batch_size = opt.batch_size if opt.mode == 'train' else opt.test_batch

        # Set the RNN model structure
        self.word_embeddings = nn.Embedding(self.vocab_size, self.input_size)
        self.RNN = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # Initialize the weights
        self.init_weights()

    def set_input(self, input):
        """load input data from the dataloader.

        Parameters:
            input: includes the input data.
        """
        pass

    def init_weights(self):
        init_range = 0.1
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

        return hidden

    def forward(self, input, hidden):
        embeddings = self.word_embeddings(input)
        output, hidden = self.RNN(embeddings, hidden)
        output = self.linear(output)

        return output, hidden