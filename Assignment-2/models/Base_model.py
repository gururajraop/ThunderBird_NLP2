"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """"
    This class is an abstract base class (ABC) for Language Models used in the project.
    """

    def __init__(self, opt):
        """
        Basic abstract model initializer
        """
        self.opt = opt

    @abstractmethod
    def set_input(self, input):
        """load input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: includes the input data.
        """
        pass

    @abstractmethod
    def train(self):
        """Training for the model"""
        pass

    @abstractmethod
    def test(self):
        """Testing of the model"""
        pass
