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
import importlib

from options import Options
from data import create_dataset
from models import create_model
import plot_graphs


if __name__ == '__main__':
    # Parse the arguments
    opt = Options().parse()

    # create a dataset given the options
    dataset = create_dataset(opt)
    vocab_size = len(dataset.vocabulary)

    # create a model given the options
    model = create_model(opt, vocab_size)

    model_filename = "models." + opt.model + "_util"
    modellib = importlib.import_module(model_filename)

    if opt.mode == 'train':
        lr = opt.lr
        train_losses = [10.00]
        train_perplexities = [100.00]
        val_losses = [10.00]
        val_perplexities = [100.00]
        prev_val_loss = 10
        for epoch in range(opt.epochs):
            loss, ppl = modellib.train_model(model, dataset, epoch + 1, lr, opt)
            train_losses.append(loss)
            train_perplexities.append(ppl)

            with open(opt.checkpoints_dir + opt.model + str(epoch + 1) + '.pt', 'wb') as f:
                torch.save(model, f)
            f.close()

            val_loss, ppl = modellib.validate_model(model, dataset, epoch + 1, opt)
            val_losses.append(val_loss)
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

            if (prev_val_loss - val_loss) < 0.1:
                lr = lr * opt.lr_decay

            prev_val_loss = val_loss

    else:
        with open(opt.checkpoints_dir + opt.model + str(opt.load_epoch) + '.pt', 'rb') as f:
            model = torch.load(f)
            model.RNN.flatten_parameters()
        f.close()

        loss, ppl = modellib.test_model(model, dataset, 1, opt)

        modellib.generate_sentences(model, dataset, sentence_len=200)


