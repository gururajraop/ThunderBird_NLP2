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
class IBM2Model(BaseModel):
    def __init__(self, opt, dataset):
        """Initialize the IBM2 class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.dataset = dataset

        self.eng_vocab = dataset.get_eng_vocabulary()
        self.french_vocab = dataset.get_french_vocabulary()

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

    #Results
    def get_perplexity(self, dataset):
        perplexity = 0.0
        for (source_sent, target_sent) in dataset.data[:100]:
            log_sent = 0.0
            for s in source_sent:
                log_sent += np.log(np.sum(list(self.prob[s].values())))

            perplexity += log_sent

        return perplexity

    def get_NLL(self, dataset, predictions):
        NLL = []
        for n, (source_sent, target_sent) in enumerate(dataset.data):
            prediction = predictions[n]
            log_likelihood = 0.0
            for _, s_idx, t_idx in prediction:
                log_likelihood += np.log(self.prob[source_sent[s_idx]][target_sent[t_idx]])
            NLL.append(log_likelihood)

        return np.mean(NLL)

    def get_best_alignments(self, dataset, epoch):
        f = open("Save/prediction_{}_{}.txt".format(self.opt.model, epoch), "w")
        alignments = []
        for n, (source_sent, target_sent) in enumerate(dataset.val_data):
            alignment = []
            for t_idx, t in enumerate(target_sent):
                best_prob = 0.0
                best_pos = 0
                for s_idx, s in enumerate(source_sent):
                    if self.prob[s][t] > best_prob:
                        best_prob = self.prob[s][t]
                        best_pos = s_idx
                # if best_pos != 0:
                alignment.append((n + 1, best_pos, t_idx))
                if self.opt.mode == 'train':
                    f.write("{} {} {} {} \n".format(n + 1, best_pos, t_idx + 1, "S"))
                else:
                    f.write("{} {} {} {} \n".format(str.zfill(n + 1, 4), best_pos, t_idx + 1, "S"))
            alignments.append(alignment)
        f.close()

        return alignments

    def get_aer(self, predictions):
        gold_sets = aer.read_naacl_alignments("datasets/validation/dev.wa.nonullalign")
        metric = aer.AERSufficientStatistics()

        for gold, pred in zip(gold_sets, predictions):
            prediction = set([(alignment[1], alignment[2]) for alignment in pred])
            metric.update(sure=gold[0], probable=gold[1], predicted=prediction)

        return metric.aer()


    def train(self, dataset, epoch):
        """Training (EM method) for the model"""

        start = time.time()
        self.EM_method(dataset)
        perplexity = self.get_perplexity(dataset)
        alignments = self.get_best_alignments(dataset, epoch)
        nll = self.get_NLL(dataset, alignments)
        aer = self.get_aer(alignments)
        print("Epoch:", epoch, ", NLL:", nll, ", Perplexity:", perplexity, ", AER:", aer, ", Total time:", int(time.time() - start), " seconds")

        return self.prob, self.gamma, perplexity, nll, aer


    def test(self):
        """Testing of the model"""
        pass