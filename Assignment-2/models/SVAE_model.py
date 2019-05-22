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
    def __init__(self, opt, vocab_size):
        """Initialize the Sentence VAE Language Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(SVAEModel, self).__init__()

        self.opt = opt
        self.type = opt.RNN_type
        self.vocab_size = vocab_size
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.latent_size = opt.latent_size

        self.batch_size = opt.batch_size if opt.mode == 'train' else opt.test_batch

        # Set the RNN model structure
        self.word_embeddings = nn.Embedding(self.vocab_size, self.input_size)

        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.decoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        hidden_factor = self.hidden_size * self.num_layers
        self.hidden2mean = nn.Linear(in_features=hidden_factor, out_features=self.latent_size)
        self.hidden2logv = nn.Linear(in_features=hidden_factor, out_features=self.latent_size)
        self.latent2hidden = nn.Linear(in_features=self.latent_size, out_features=hidden_factor)

        self.outputs2vocab = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

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
        self.hidden2mean.weight.data.uniform_(-init_range, init_range)
        self.hidden2mean.bias.data.zero_()
        self.hidden2logv.weight.data.uniform_(-init_range, init_range)
        self.hidden2logv.bias.data.zero_()
        self.latent2hidden.weight.data.uniform_(-init_range, init_range)
        self.latent2hidden.bias.data.zero_()
        self.outputs2vocab.weight.data.uniform_(-init_range, init_range)
        self.outputs2vocab.bias.data.zero_()

    def forward(self, input, batch_size):
        embeddings = self.word_embeddings(input)
        _, hidden = self.encoder(embeddings)
        hidden = hidden.view(batch_size, self.hidden_size * self.num_layers)

        # Reparameterization trick
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        # Generate the latent space
        z = torch.randn([batch_size, self.latent_size])
        z = z * std + mean
        hidden = self.latent2hidden(z)
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)

        # decoder
        output, _ = self.decoder(embeddings, hidden)
        logp = self.outputs2vocab(output)

        return logp, mean, logv, z