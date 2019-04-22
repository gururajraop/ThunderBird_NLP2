"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import torch
import numpy as np
from collections import defaultdict

from .Base_model import BaseModel

class IBM1Model(BaseModel):
    def __init__(self, opt, dataset):
        """Initialize the IBM1 class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.t = self.initialize_probabilities(dataset)

    def set_input(self, input):
        """load input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the input data dictionary.
        """
        pass

    def train(self, dataset):
        """Training (EM method) for the model"""
        print(self.t)


    def test(self):
        """Testing of the model"""
        pass

    def initialize_probabilities(self, dataset):
        """Uniformaly initialize all the probabilities"""
        t = defaultdict(float)

        return t
