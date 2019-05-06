"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
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
        parser.add_argument('--dataroot', default='./datasets/', help='path to input (both training and testing)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Base path to save or load the trained checkpoints')
        parser.add_argument('--mode', type=str, default='train', help='Training or Testing mode. Options: [train | test]')

        self.initialized = True
        self.isTrain = False
        return parser

    def gather_options(self):
        """
        Gathers the options specified and parses them to a suitable format
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.isTrain = True if opt.mode == "train" else False

        # save and return the parser
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt

        return self.opt