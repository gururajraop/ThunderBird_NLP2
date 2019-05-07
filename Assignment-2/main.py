"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
import sys
import numpy as np

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

    if opt.mode == 'train':
        model.train(dataset)
    else:
        model.test(dataset)