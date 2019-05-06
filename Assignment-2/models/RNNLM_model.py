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


class RNNLMModel(BaseModel):
    def __init__(self, opt, dataset):
        """Initialize the Deterministic RNN Language Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt

        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.output_size = opt.output_size

        #
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        if opt.RNN_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        elif opt.RNN_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            assert False, "Error! Wrong type of RNN model for the RNNLM"
        self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)
        self.SoftMax = nn.Softmax()



    def set_input(self, input):
        """load input data from the dataloader.

        Parameters:
            input: includes the input data.
        """
        pass

    def train(self):
        """Training for the model"""
        pass

    def test(self):
        """Testing of the model"""
        pass