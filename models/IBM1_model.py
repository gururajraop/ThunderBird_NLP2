"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import torch
import numpy as np
from collections import defaultdict
import time
import aer

from .Base_model import BaseModel

WORST_PERPLEXITY = -100
WORST_NLL = 250
WORST_AER = 1.0


class IBM1Model(BaseModel):
    def __init__(self, opt, dataset):
        """Initialize the IBM1 class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt

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
        """Perform the Expectation Maximization step for IBM model 1

        Parameters:
            dataset: The training dataset
        """
        print("Performing Parameter Optimization using EM method")
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)

        # E-Step
        # print("E-Step")
        for (source_sent, target_sent) in dataset.data:
            for t in set(target_sent):
                norm = 0.0
                for s in set(source_sent):
                    norm += self.prob[s][t] * target_sent.count(t)
                for s in set(source_sent):
                    denom = (self.prob[s][t] * source_sent.count(s) * target_sent.count(t)) / norm
                    count[s][t] += denom
                    total[s] += denom

        # M-Step
        # print("M-Step")
        for s in count.keys():
            for t in count[s].keys():
                self.prob[s][t] = count[s][t] / total[s]

    def get_perplexity(self, dataset):
        """Compute the perplexity of the model using the probabilities
        Here the perplexity is measured in terms of Cross Entropy
        Perplexity = - (1/N) \Sigma_k=1^N log(P(f^(k)|e^(k)))

        Parameters:
            dataset: The training dataset
        Returns:
            Perplexity
        """
        print("Computing Perplexity for the training data")
        perplexity = []
        for (source_sent, target_sent) in dataset.val_data:
            log_sent = 0.0
            for s in source_sent:
                prob_list = list(self.prob[s].values())
                log_sent += np.log(max(prob_list)) if len(prob_list) != 0 else 0

            perplexity.append(log_sent)

        if np.sum(perplexity) == 0:
            perplexity = [WORST_PERPLEXITY]

        return np.mean(perplexity)

    def get_NLL(self, dataset, epoch):
        """Compute the Negative Log-Likelihood of the model using the probabilities
        NLL = argmin -\Sigma_k=1^N (1/N) log(P(f^(k)|e^(k)))

        Parameters:
            dataset: The training dataset
            epoch:   The current epoch
        Returns:
            Negative Log-Likelihood
        """
        print("Computing NLL for the training data")
        NLL = []
        predictions = self.get_best_alignments(dataset.val_data, epoch)
        for n, (source_sent, target_sent) in enumerate(dataset.val_data):
            prediction = predictions[n]
            log_likelihood = 0.0
            for (_, s_idx, t_idx) in prediction:
                log_likelihood += np.log(self.prob[source_sent[s_idx]][target_sent[t_idx-1]])
            NLL.append(-log_likelihood)

        return np.mean(NLL)

    def get_best_alignments(self, data, epoch):
        """Find the best alignments of the model using the probabilities

        Parameters:
            dataset: The training dataset
            epoch:   The current epoch
        Returns:
            Best alignments for each sentence
        """
        print("Obtaining the best alignments")
        if self.opt.mode == 'test':
            f = open("Save/prediction_{}_{}.txt".format(self.opt.model, epoch), "w")

        alignments = []
        for n, (source_sent, target_sent) in enumerate(data):
            alignment = []
            for t_idx, t in enumerate(target_sent):
                best_prob = 0.0
                best_pos = 0
                for s_idx, s in enumerate(source_sent):
                    if self.prob[s][t] > best_prob:
                        best_prob = self.prob[s][t]
                        best_pos = s_idx

                alignment.append((n+1, best_pos, t_idx+1)) #Skip the NULL character
                if self.opt.mode == 'test':
                    f.write("{} {} {} {} \n".format(n+1, best_pos, t_idx+1, "S"))
            alignments.append(alignment)
        if self.opt.mode == 'test':
            f.close()

        return alignments

    def get_aer(self, dataset, epoch):
        """Compute the Alignment Error Rate of the model using the best alignments

        Parameters:
            dataset: The training dataset
            epoch:   The current epoch
        Returns:
            AER score
        """
        print("Computing AER on validation dataset")
        gold_sets = aer.read_naacl_alignments("datasets/validation/dev.wa.nonullalign")
        metric = aer.AERSufficientStatistics()

        predictions = self.get_best_alignments(dataset.val_data, epoch)

        for gold, pred in zip(gold_sets, predictions):
            prediction = set([(alignment[1], alignment[2]) for alignment in pred])
            metric.update(sure=gold[0], probable=gold[1], predicted=prediction)

        return metric.aer()

    def train(self, dataset, epoch):
        """Training (EM method) for the model"""

        start = time.time()
        self.EM_method(dataset)
        perplexity = self.get_perplexity(dataset)
        nll = self.get_NLL(dataset, epoch)
        aer = self.get_aer(dataset, epoch)
        print("Epoch:", epoch, ", NLL:", nll, ", Perplexity:", perplexity, ", AER:", aer, ", Total time:", int(time.time() - start), " seconds")

        #None to accomodate gamma probability for IBM2
        return self.prob, None, perplexity, nll, aer

    def test(self):
        """Testing of the model"""
        pass

    def initialize_probabilities(self, vocab_size):
        """Uniformaly initialize all the probabilities"""

        prob = defaultdict(lambda: defaultdict(lambda: 1 / vocab_size))

        return prob
