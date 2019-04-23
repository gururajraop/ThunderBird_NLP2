"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import torch
import numpy as np
from collections import defaultdict
import time

from .Base_model import BaseModel

class IBM1Model(BaseModel):
    def __init__(self, opt, dataset):
        """Initialize the IBM1 class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.eng_vocab = dataset.get_eng_vocabulary()
        self.french_vocab = dataset.get_french_vocabulary()

        self.prob = self.initialize_probabilities(self.french_vocab)

    def set_input(self, input):
        """load input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the input data dictionary.
        """
        pass

    def EM_method(self, dataset):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)

        # E-Step
        # print("E-Step")
        for (f_sent, e_sent) in dataset.data:
            for e in e_sent:
                total_sen = 0.0
                for f in f_sent:
                    total_sen += self.prob[e][f]
                for f in f_sent:
                    count[e][f] += self.prob[e][f] / total_sen
                    total[f] += self.prob[e][f] / total_sen

        # M-Step
        # print("M-Step")
        for e in count.keys():
            for f in count[e].keys():
                self.prob[e][f] = count[e][f] / total[f]

    def get_NLL(self):
        max_total = 0.0
        for e in self.prob.keys():
            max_total += max(list(self.prob[e].values()))

        nll = - np.log(max_total / len(self.prob.keys()))
        return nll


    def train(self, dataset, epoch):
        """Training (EM method) for the model"""
        # print("----------------Before----------------")
        # print(self.prob['Session']['Session'])

        start = time.time()
        self.EM_method(dataset)
        nll = self.get_NLL()
        print("Epoch:", epoch, ", NLL:", nll, ", Total time:", time.time() - start, " seconds")

        # print("----------------After----------------")
        # print(self.prob['Session']['Session'])


    def test(self):
        """Testing of the model"""
        pass

    def initialize_probabilities(self, vocab):
        """Uniformaly initialize all the probabilities"""

        vocab_len = len(vocab)
        prob = defaultdict(lambda: defaultdict(lambda: 1 / vocab_len))
        # prob = defaultdict(lambda : 1 / vocab_len)

        return prob
