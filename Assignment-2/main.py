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
import plot_graphs


def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(detach_hidden(h) for h in hidden)


def train_model(model, dataset, epoch, lr, opt):
    print("-----------------------------------Training-----------------------------------")
    model.train()
    total_loss = []
    hidden = model.init_hidden(opt.batch_size)
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
            start = time.time()


    print('\nEpoch: {:5d} | Average loss: {:5.4f} | Average Perplexity : {:5.4f}'.format(
        epoch, np.mean(total_loss), np.mean(perplexity)))

    return np.mean(total_loss), np.mean(perplexity)


def validate_model(model, dataset, epoch, opt):
    print("----------------------------------Validation----------------------------------")
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(opt.test_batch)
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.val_data)
    start = time.time()
    perplexity = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, dataset.val_data.size(0) - 1, opt.seq_length)):
            source, target = dataset.load_data('val', idx)
            hidden = detach_hidden(hidden)
            output, hidden = model(source, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion_loss(output, target)
            total_loss += len(source) * loss.item()

            perplexity += np.exp(loss.item()) * len(source)

    loss = total_loss / data_size
    per_word_ppl = perplexity / vocab_size
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, elapsed_time))

    return loss, per_word_ppl


def test_model(model, dataset, epoch, opt):
    print("----------------------------------Testing----------------------------------")
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(opt.test_batch)
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.test_data)
    start = time.time()
    perplexity = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, dataset.test_data.size(0) - 1, opt.seq_length)):
            source, target = dataset.load_data('test', idx)
            hidden = detach_hidden(hidden)
            output, hidden = model(source, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion_loss(output, target)
            total_loss += len(source) * loss.item()

            perplexity += np.exp(loss.item()) * len(source)

    loss = total_loss / data_size
    per_word_ppl = perplexity / vocab_size
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, elapsed_time))

    return loss, per_word_ppl


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
        train_losses = [10.00]
        train_perplexities = [2000.00]
        val_losses = [10.00]
        val_perplexities = [2000.00]
        for epoch in range(opt.epochs):
            loss, ppl = train_model(model, dataset, epoch + 1, lr, opt)
            train_losses.append(loss)
            train_perplexities.append(ppl)

            with open(opt.checkpoints_dir + opt.model + str(epoch + 1) + '.pt', 'wb') as f:
                torch.save(model, f)
            f.close()

            loss, ppl = validate_model(model, dataset, epoch + 1, opt)
            val_losses.append(loss)
            val_perplexities.append(ppl)

            losses = (train_losses, val_losses)
            perplexities = (train_perplexities, val_perplexities)

            title = 'Losses as function of iteration'
            save_path = opt.log_dir + opt.model + '_Loss_' + str(epoch + 1) + '.png'
            legend = ['training loss', 'validation loss']
            plot_graphs.plot(losses, epoch+1, 'loss', title, legend, save_path)
            title = 'Perplexity as function of iteration'
            save_path = opt.log_dir + opt.model + '_PPL_' + str(epoch + 1) + '.png'
            legend = ['training perplexity', 'validation perplexity']
            plot_graphs.plot(perplexities, epoch+1, 'ppl', title, legend, save_path)

            lr = lr / 2
    else:
        with open(opt.checkpoints_dir + opt.model + str(opt.load_epoch) + '.pt', 'rb') as f:
            model = torch.load(f)
            model.RNN.flatten_parameters()
        f.close()

        loss, ppl = test_model(model, dataset, 1, opt)
