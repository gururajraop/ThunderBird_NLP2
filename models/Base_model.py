"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """"
    This class is an abstract base class (ABC) for both IBM1 and IBM2 models.
    """

    def __init__(self, opt):
        """

        :param opt:
        """
        self.opt = opt

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def train(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass
