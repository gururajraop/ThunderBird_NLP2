"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
import sys
import numpy as np
import torch
from torch import nn
import time

from options import Options
from data import create_dataset
from models import create_model


def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(detach_hidden(h) for h in hidden)


def train_model(model, dataset, epoch, lr, opt):
    model.train()
    total_loss = []
    hidden = model.init_hidden()
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.train_data)
    start = time.time()
    perplexity = []

    for batch, idx in enumerate(range(0, dataset.train_data.size(0) - 1, opt.seq_length)):
        source, target = dataset.load_data('train', idx)
        hidden = detach_hidden(hidden)
        model.zero_grad()
        output, hidden = model(source, hidden)
        loss = criterion_loss(output.view(-1, vocab_size), target)
        loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss.append(loss.item())
        ppl = np.exp(loss.item()) / opt.seq_length
        perplexity.append(ppl)

        if (batch % opt.print_interval == 0) and batch != 0:
            elapsed_time = (time.time() - start) * 1000 / opt.print_interval
            print('Epoch: {:5d} | {:5d}/{:5d} batches | LR: {:5.4f} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
                epoch, batch, data_size // opt.seq_length, lr, np.mean(total_loss), np.mean(perplexity), elapsed_time))
            total_loss = []
            start = time.time()


if __name__ == '__main__':
    # Parse the arguments
    opt = Options().parse()

    # create a dataset given the options
    dataset = create_dataset(opt)
    vocab_size = len(dataset.vocabulary)

    # create a model given the options
    model = create_model(opt, vocab_size)

    if opt.mode == 'train':
        lr = opt.lr
        for epoch in range(opt.epochs):
            train_model(model, dataset, epoch + 1, lr, opt)
            lr = lr / 5
    else:
        model.test(dataset)