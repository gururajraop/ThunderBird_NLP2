"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import dill

from options import Options
from data import create_dataset
from models import create_model


def plot_chart(values, epoch, legend, xlabel, ylabel, title, save):
    x = [i for i in range(epoch + 1)]
    plt.plot(x, values, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.savefig(save)
    plt.close()


if __name__ == '__main__':
    # Parse the arguments
    opt = Options().parse()

    # create a dataset given the options
    dataset = create_dataset(opt)

    # create a model given the options
    model = create_model(opt, dataset)

    # Run the training for the model
    if opt.mode == 'train':
        perplexity_values = [model.get_perplexity(dataset)]
        nll_values = [model.get_NLL(dataset)]
        aer_values = [model.get_aer()]
        print("Epoch: 0  NLL:", nll_values[0], ", Perplexity:", perplexity_values[0], ", AER:", aer_values[0], ", Total time:0.0  seconds")
        for epoch in range(opt.epoch, opt.n_iters+1):
            prob, perplexity, nll, aer = model.train(dataset, epoch)
            perplexity_values.append(perplexity)
            nll_values.append(nll)
            aer_values.append(aer)

            # Save the model
            with open('Save/IBM1_' + str(epoch) + '.pkl', 'wb') as f:
                dill.dump(prob, f, pickle.HIGHEST_PROTOCOL)

            # Save the various progress chart
            title = "Training Log-Likelihood (Perplexity) as a function of iterations"
            save = 'Save/IBM1_Perplexity_' + str(epoch) + '.png'
            plot_chart(perplexity_values, epoch, "Perplexity", "Iteration-->", "Perplexity-->", title, save)

            title = "Training Log-Likelihood (NLL) as a function of iterations"
            save = 'Save/IBM1_NLL_' + str(epoch) + '.png'
            plot_chart(nll_values, epoch, "NLL", "Iteration-->", "NLL-->", title, save)

            title = "Training AER values as a function of iterations"
            save = 'Save/IBM1_AER_' + str(epoch) + '.png'
            plot_chart(aer_values, epoch, "AER", "Iteration-->", "AER-->", title, save)
    else:
        model.test(dataset)
