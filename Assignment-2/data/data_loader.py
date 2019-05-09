"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

from nltk.tree import Tree
from collections import defaultdict
import torch

class Data:
    """
    The base data that holds the vocabulary and the tokenized data
    """
    def __init__(self):
        """"Initialization of the vocabulary and the word token"""
        self.vocab = []
        self.word2token = {}

    def add_word(self, word):
        """"Add a new word to the vocabulary and tokenizer"""
        if word not in self.vocab:
            self.vocab.append(word)
            self.word2token[word] = len(self.vocab) - 1

        return self.word2token[word]

    def __len__(self):
        """Returns the length of the vocabulary"""
        return len(self.vocab)

class DataLoader():
    """
    The data loading class
    This class handles the data loading and pre-processing of the data
    """

    def __init__(self, opt):
        """
        Initialization fo the DataLoader
        """
        self.train_batch_size = opt.batch_size
        self.test_batch_size = opt.test_batch
        self.seq_len = opt.seq_length

        self.vocabulary = Data()

        if opt.mode == 'train':
            self.mode = 'train'
            self.train_data_path = opt.dataroot+"/Training/02-21.10way.clean"
            self.val_data_path = opt.dataroot + "/Validation/22.auto.clean"

            print("Processing Training data")
            self.train_data = self.get_data(self.train_data_path)
            self.train_data = self.get_batched_data('train')

            print("Processing Validation data")
            self.val_data = self.get_data(self.val_data_path)
            self.val_data = self.get_batched_data('val')
        else:
            self.mode = 'test'
            self.test_data_path = opt.dataroot+"/Testing/23.auto.clean"

            print("Processing Testing data")
            self.test_data = self.get_data(self.test_data_path)
            self.test_data = self.get_batched_data('test')

    def get_data(self, data_path):
        """
        Obtain the parsed data using the provided data file path

        Parameters:
            data_path  : Path to the data file

        Returns:
            The parsed data in a list format
        """
        file = open(data_path, 'r', encoding='utf8')
        lines = file.readlines()

        tokens = []
        for line in lines:
            sentence = self.pre_process_data(line)
            for word in sentence:
                self.vocabulary.add_word(word)
                tokens.append(self.vocabulary.word2token[word])

        data = torch.LongTensor(tokens)

        file.close()

        return data

    def pre_process_data(self, line):
        """
        Parse the line in the semantic tree structure and get the sentence
        Also, performs few pre-processing to get rid of unwanted strings

        Parameters:
            line:   Unparsed raw line from the file

        Returns:
            Parsed line with some pre-processing
        """
        tree = Tree.fromstring(line)
        words = tree.leaves()

        # Remove some unwanted characters such as punctuation marks and special characters
        unwanted = ['``', ',', '\'\'', '&', '.', '!', '?', '@', '#', '$', '%', '\"\"']
        sentence = [word for word in words if word not in unwanted]

        # Add the Start-Of-Sentence (SOS) for the sentence
        sentence = ['-SOS-'] + sentence

        return sentence

    def load_data(self, mode, index):
        """
        Return the data from a specific index
        """
        if mode == 'train':
            data = self.train_data
        elif mode == 'val':
            data = self.val_data
        else:
            data = self.test_data

        seq_len = min(self.seq_len, len(data) - 1 - index)
        source = data[index:index+seq_len].view(-1)
        target = data[index+1:index+1+seq_len].view(-1)

        return source, target

    def get_batched_data(self, mode):
        """Divide the data to batched chunks"""
        if mode == 'train':
            data = self.train_data
            batch_size = self.train_batch_size
        elif mode == 'val':
            data = self.val_data
            batch_size = self.test_batch_size
        else:
            data = self.test_data
            batch_size = self.test_batch_size

        num_batches = data.size(0) // batch_size
        data = data.narrow(0, 0, num_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()

        return data
