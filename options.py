"""
University of Amsterdam: MSc AI: NLP2 2019 Spring

Project 1: Lexical Alignment

Team 3: Gururaja P Rao, Manasa J Bhat

Author: Gururaja P Rao
"""

import argparse
import os

class Options:
    """This class defines options used during both training and test time.

    It also implements few helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to input (both training and testing)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Base path to save or load the trained checkpoints')
        parser.add_argument('--name', required=True, help='Sub path to saved checkpoints (both training and testing)')
        parser.add_argument('--mode', type=str, default='test', help='Sub path to saved checkpoints (both training and testing)')

        # model parameters
        parser.add_argument('--model', type=str, default='IBM1', help='chooses which model to use. Options: [IBM1 | IBM2]')

        # dataset parameters
        parser.add_argument('--english', type=str, default='e', help='Suffix of english filename')
        parser.add_argument('--french', type=str, default='f', help='Suffix of french filename')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--max_sentences', type=int, default=10000, help='Maximum number of sentences to use for training')

        # additional parameters
        parser.add_argument('--epoch', type=int, default=0, help='The starting epoch. If 0 the model will be intialized freshly, else the model will be loaded by the checkpoint based on the epoch')
        self.initialized = True
        return parser

    def gather_options(self):
        """
        Gathers the options specified and parses them to a suitable format
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        return parser.parse_known_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt, _ = self.gather_options()

        self.print_options(opt)

        self.opt = opt

        return self.opt
