"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat

Author: Gururaja P Rao
"""

import os
import sys
import numpy as np

from options import Options
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = Options().parse()

    dataset = create_dataset(opt)  # create a dataset given the options

    model = create_model(opt)      # create a model given the options