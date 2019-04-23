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

    def EM_method(self, dataset):
        count = defaultdict(float)
        total = defaultdict(float)
        # total_sen = defaultdict(float)

        # E-Step
        for (f_sent, e_sent) in dataset.data:
            for e in e_sent:
                # total_sen[e] = 0.0
                total_sen = 0.0
                for f in f_sent:
                    # total_sen[e] += self.t[(e, f)]
                    total_sen += self.t[(e, f)]
                for f in f_sent:
                    count[(e, f)] += self.t[(e, f)] / total_sen
                    total[f] += self.t[(e, f)] / total_sen

            """
            for e in e_sent:
                for f in f_sent:
                    count[(e, f)] += self.t[(e, f)] / total_sen[e]
                    total[f] += self.t[(e, f)] / total_sen[e]
            """

        # M-Step
        for f in set(total):
            for e in set(total_sen):
                self.t[(e, f)] = count[(e, f)] / total[f]


    def train(self, dataset):
        """Training (EM method) for the model"""
        print("----------------Before----------------")
        print(self.t['Session', 'Session'])

        self.EM_method(dataset)

        print("----------------After----------------")
        print(self.t['Session', 'Session'])


    def test(self):
        """Testing of the model"""
        pass

    def initialize_probabilities(self, dataset):
        """Uniformaly initialize all the probabilities"""

        vocab_len = len(dataset.data)
        t = defaultdict(lambda : len(vocab_len))

        return t
