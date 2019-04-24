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

if __name__ == '__main__':
    # Parse the arguments
    opt = Options().parse()

    # create a dataset given the options
    dataset = create_dataset(opt)

    # create a model given the options
    model = create_model(opt, dataset)

    # Run the training for the model
    if opt.mode == 'train':
        nll_values = [model.get_NLL()]
        print("Epoch: 0 , NLL:", nll_values[0], ", Total time: 0.0  seconds")
        for epoch in range(opt.epoch, opt.n_iters+1):
            prob, nll = model.train(dataset, epoch)
            nll_values.append(nll)

            # Save the model
            with open('Save/IBM1_' + str(epoch) + '.pkl', 'wb') as f:
                dill.dump(prob, f, pickle.HIGHEST_PROTOCOL)

            # Save the nll progress chart
            x = [i for i in range(epoch+1)]
            plt.plot(x, nll_values, label='NLL')
            plt.xlabel('Iteration -->')
            plt.ylabel('NLL values -->')

            plt.title("Training Log-Likelihood as a function of iterations")

            plt.legend()
            plt.grid(True)
            plt.savefig('Save/IBM1_NLL_' + str(epoch) + '.png')
            plt.close()
    else:
        model.test(dataset)
