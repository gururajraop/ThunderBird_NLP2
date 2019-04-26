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
import dill
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
        self.prob, self.gamma = self.initialize_probabilities(self.french_vocab, type=opt.init_type)


    def set_input(self, input):
        """load input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the input data dictionary.
        """
        pass

    def jump(self, i, j, l, m):
        idx = i - np.floor(j * l / m)
        return idx

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
            m = len(target_sent)
            l = len(source_sent)
            for i, t in enumerate(target_sent):
                norm = 0.0
                for j, s in enumerate(source_sent):
                    x = self.jump(i, j, l, m)
                    gamma_st[s][t] = self.gamma[x]
                    norm += self.prob[s][t] * self.gamma[x]
                for j, s in enumerate(source_sent):
                    norm = 1e-6 if norm == 0 else norm
                    weight = (self.prob[s][t] * gamma_st[s][t]) / norm
                    count[s][t] += weight
                    total[t] += weight
                    n_Yx[gamma_st[s][t]] += weight

        # M-Step
        # print("M-Step")
        for s in count.keys():
            for t in count[s].keys():
                self.prob[s][t] = count[s][t] / total[t]

        sum_x = sum(n_Yx.values())
        for x in n_Yx.keys():
            self.gamma[x] = n_Yx[x]/sum_x

    def initialize_probabilities(self, vocab, type = 'uniform'):
        """
        Initialize the lexical parameters of IBM model 2 using various methods
        Parameters:
            vocab:  vocabulary
            type:   Initialization type

        Returns:
            Initialized lexical and alignment parameters
        """
        if type == 'uniform':
            """Uniformaly initialize all the probabilities"""
            print("Uniform initialization of the IBM2 model parameters")
            vocab_len = len(vocab)
            prob = defaultdict(lambda: defaultdict(lambda: 1 / vocab_len))
            init = 1/(2 * self.L + 1)
            gamma = defaultdict(lambda: init)
        elif type == 'random':
            """Randomly initialize all the probabilities"""
            print("Random initialization of the IBM2 model parameters")
            prob = defaultdict(lambda: defaultdict(lambda: np.random.rand()))
            gamma = defaultdict(lambda: np.random.randint(-self.L, self.L))
        elif type == 'IBM1':
            """IBM1 initialization all the probabilities"""
            print("IBM1 initialization of the IBM2 model parameters")
            with open('Save/IBM1_10.pkl', 'rb') as in_strm:
                model_1 = dill.load(in_strm)
            prob = model_1.prob
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
            #f = open("Save/prediction_{}_{}.txt".format(self.opt.model, epoch), "w")
            f = open("Save/ibm2.mle.naacl", "w")
            print("Writing NACCL files")

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
                if best_pos != 0:
                    alignment.append((n+1, best_pos, i+1)) #Skip the NULL character
                if self.opt.mode == 'test':
                    if best_prob > 0.5:
                        f.write("{} {} {} {} \n".format(str(n+1).zfill(4), best_pos, i+1, "S"))
                    else:
                        f.write("{} {} {} {} \n".format(str(n+1).zfill(4), best_pos, i+1, "P"))
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

    def test(self, dataset):
        """Testing of the model"""
        """Testing of the model"""
        aer = self.get_aer(dataset, 10)
        print("AER score on testing dataset:", aer)
