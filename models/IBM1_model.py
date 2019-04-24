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

        if opt.direction == "E2F":
            self.source_len, self.target_len = len(self.eng_vocab), len(self.french_vocab)
        else:
            self.source_len, self.target_len = len(self.french_vocab), len(self.eng_vocab)

        self.prob = self.initialize_probabilities(self.target_len)

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
        for (source_sent, target_sent) in dataset.data:
            for t in target_sent:
                total_sent = 0.0
                for s in source_sent:
                    total_sent += self.prob[s][t]
                for s in source_sent:
                    count[s][t] += self.prob[s][t] / total_sent
                    total[s] += self.prob[t][s] / total_sent

        # M-Step
        # print("M-Step")
        for s in count.keys():
            for t in count[s].keys():
                self.prob[s][t] = count[s][t] / total[s]

    def get_perplexity(self, dataset):
        perplexity = 0.0
        for (source_sent, target_sent) in dataset.data:
            log_sent = 0.0
            for t in target_sent:
                log_sum = []
                for s in source_sent:
                    log_sum.append(self.prob[s][t])
                log_sent += np.log(np.sum(log_sum))

        perplexity += log_sent

        return perplexity

    def get_NLL(self):
        max_total = 0.0
        print(len(self.prob.keys()))
        i = 0
        for s in self.prob.keys():
            max_total += max(list(self.prob[s].values()))
            i += 1

        print(i)
        print(max_total)

        if max_total == 0:
            nll = -np.log(1 / len(self.target_len))
        else:
            nll = -np.log(max_total / len(self.prob.keys()))

        return nll

    def get_aer(self):
        return 0.0

    def train(self, dataset, epoch):
        """Training (EM method) for the model"""

        start = time.time()
        self.EM_method(dataset)
        perplexity = self.get_perplexity(dataset)
        nll = self.get_NLL()
        aer = self.get_aer()
        print("Epoch:", epoch, ", NLL:", nll, ", Perplexity:", perplexity, ", AER:", aer, ", Total time:", int(time.time() - start), " seconds")

        return self.prob, perplexity, nll, aer

    def test(self):
        """Testing of the model"""
        pass

    def initialize_probabilities(self, vocab_size):
        """Uniformaly initialize all the probabilities"""

        prob = defaultdict(lambda: defaultdict(lambda: 1 / vocab_size))

        return prob
