"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

from .data_loader import DataLoader

def create_dataset(opt):
    """Create the dataset
    """
    dataset = DataLoader(opt)
