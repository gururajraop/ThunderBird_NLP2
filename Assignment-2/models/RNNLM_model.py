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

        self.type = opt.RNN_type
        self.vocab_size = opt.vocab_size
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.output_size = opt.output_size
        self.batch_size = opt.batch_size

        self.drop = nn.Dropout(0.5)

        # Set the RNN model structure
        self.word_embeddings = nn.Embedding(self.vocab_size, self.input_size)
        if opt.RNN_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True
            )
        elif opt.RNN_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            assert False, "Error! Wrong type of RNN model for the RNNLM"
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.SoftMax = nn.Softmax()

        self.criterion_loss = nn.CrossEntropyLoss()

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

    def init_hidden(self):
        weight = next(self.parameters)
        weight.new_zeros(self.num_layers, self.batch_size, self.hidden_size)

        return weight

    def forward(self, input, hidden):
        embeddings = self.word_embeddings(input)
        output, hidden = self.RNN(embeddings, hidden)
        output = self.drop(output)
        pred = self.linear(output.view(output.size(0)*output.size(1), output.size(2)))
        pred = pred.view(output.size(0), output.size(1), pred.size(1))

        return pred, hidden

    def get_batch_data(self, data, batch_size):
        X = []
        y = []
        for i in range(0, len(data), batch_size):
            single_batch = np.array(data[i:i+batch_size])
            single_batch_x = np.delete(single_batch, -1, axis=1)
            single_batch_y = np.delete(single_batch, 0, axis=1)
            X.append(single_batch_x) #total_batches * batch_size * input_size
            y.append(single_batch_y) #total_batches * batch_size * targets_for_each_time_step(=input_size)

        return X, y

    def train(self, dataset):
        """Training for the model"""
        input_batches, target_batches = self.get_batch_data(dataset.tokenized_train_data, self.batch_size)
        for index, batch in enumerate(input_batches):
            #Each input is batch_size * input_size
            input = torch.Tensor(batch).long()
            target = torch.Tensor(target_batches[index]).long()

            # call init_hidden after fixed. Gives some error now.
            hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                      torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

            pred, hidden = self.forward(input, hidden)
            loss = self.criterion_loss(pred.view(-1, self.vocab_size), target.view(target.shape[0] * target.shape[1]))
            print(index, loss.item())
            loss.backward()


    def test(self, dataset):
        """Testing of the model"""
        pass