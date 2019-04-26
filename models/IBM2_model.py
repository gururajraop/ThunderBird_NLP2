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
import pickle
import aer

WORST_PERPLEXITY = -100
WORST_NLL = 250
WORST_AER = 1.0


class IBM2Model(BaseModel):
    def __init__(self, opt, dataset):
        """Initialize the IBM2 class.

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

        self.L, self.M = self.get_max_sentence_length(dataset)
        self.prob, self.gamma = self.initialize_probabilities(self.french_vocab)


    def set_input(self, input):
        """load input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the input data dictionary.
        """
        pass

    def jump(self, i, j, l, m):
        return (i - ((j * l)/m))

    def EM_method(self, dataset):
        """Perform the Expectation Maximization step for IBM model 1

        Parameters:
            dataset: The training dataset
        """
        print("Performing Parameter Optimization using EM method")
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(lambda: 1.0)
        gamma_st = defaultdict(lambda: defaultdict(float))
        n_Yx = defaultdict(lambda: 0.0)
        for (source_sent, target_sent) in dataset.data:
            l = len(target_sent)
            m = len(source_sent)
            for j, s in enumerate(source_sent):
                norm = 0.0
                for i,t in enumerate(target_sent):
                    x = self.jump(i, j, l, m)
                    gamma_st[s][t] = self.gamma[x]
                    norm += self.prob[s][t] * self.gamma[x]
                for t in target_sent:
                    count[s][t] += (self.prob[s][t] * gamma_st[s][t]) / (norm)
                    total[t] += (self.prob[s][t] * gamma_st[s][t]) / (norm)
                    n_Yx[gamma_st[s][t]] += (self.prob[s][t] * gamma_st[s][t]) / (norm)

        # M-Step
        # print("M-Step")
        for s in count.keys():
            for t in count[s].keys():
                self.prob[s][t] = count[s][t] / total[t]

        sum_x = sum(n_Yx.values())
        for x in n_Yx.keys():
            self.gamma[x] = n_Yx[x]/sum_x

    def initialize_probabilities(self, vocab, type = 'uniform'):
        if type == 'uniform':
            """Uniformaly initialize all the probabilities"""
            vocab_len = len(vocab)
            prob = defaultdict(lambda: defaultdict(lambda: 1 / vocab_len))
            init = 1/(2 * self.L + 1)
            gamma = defaultdict(lambda: init)
        elif type == 'random':
            prob = defaultdict(lambda: defaultdict(lambda: np.random.rand()))
            gamma = defaultdict(lambda: np.random.randint(-self.L, self.L))
        elif type == 'IBM1':
            prob = pickle.load('prob.pickle')
            init = 1 / (2 * self.L + 1)
            gamma = defaultdict(lambda: init)
        return prob, gamma

    def get_max_sentence_length(self, dataset):
        eng_sentences = [s for s,t in dataset.data]
        fre_sentences = [t for s, t in dataset.data]
        L = len(max(eng_sentences, key=len))
        M = len(max(fre_sentences, key=len))
        return L, M

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
            m = len(source_sent)
            l = len(target_sent)
            for j, s in enumerate(source_sent):
                log_sum = []
                for i, t in enumerate(target_sent):
                    x = self.jump(i, j, l, m)
                    log_sum.append(self.prob[s][t] * self.gamma[x])
                log_sent += np.log(max(log_sum))

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
            m = len(source_sent)
            l = len(target_sent)
            for i, t in enumerate(target_sent):
                best_prob = 0.0
                best_pos = 0
                for j, s in enumerate(source_sent):
                    x = self.jump(i, j, l, m)
                    prob = self.prob[s][t] * self.gamma[x]
                    if prob > best_prob:
                        best_prob = prob
                        best_pos = j

                alignment.append((n+1, best_pos, i+1)) #Skip the NULL character
                if self.opt.mode == 'test':
                    f.write("{} {} {} {} \n".format(n+1, best_pos, i+1, "S"))
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

        return self.prob, self.gamma, perplexity, nll, aer


    def test(self):
        """Testing of the model"""
        pass