"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
import sys
import numpy as np
import torch
from torch import nn
import time


def train_model(model, dataset, epoch, lr, opt):
    pass


def validate_model(model, dataset, epoch, opt):
    pass


def test_model(model, dataset, epoch, opt):
    pass


def compute_accuracy(output, target):
    output = torch.argmax(output, dim=2)
    correct = (target == output).float()
    accuracy = torch.mean(correct)

    return accuracy


def generate_sentences(model, dataset, sentence_len):
    pass