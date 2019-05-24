"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

from nltk.tree import Tree
from collections import defaultdict
import torch
import dill
import pickle
import os

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
        self.max_seq_len = opt.seq_length

        if os.path.isfile(opt.dataroot + 'vocabulary.pkl'):
            print("Loading processed vocabulary from file")
            with open(opt.dataroot + 'vocabulary.pkl', 'rb') as in_file:
                self.vocabulary = dill.load(in_file)
            in_file.close()
        else:
            self.vocabulary = Data()
            special_words = ['-PAD-', '-UNK-', '-SOS-', '-EOS-']
            for word in special_words:
                self.vocabulary.add_word(word)

        if opt.mode == 'train':
            self.mode = 'train'

            # Get the processed training dataset
            if os.path.isfile(opt.dataroot + 'train_data.pkl'):
                print("Loading the processed training data from file")
                with open(opt.dataroot + 'train_data.pkl', 'rb') as in_file:
                    self.train_data = dill.load(in_file)
                in_file.close()
            else:
                print("Processing Training data")
                train_data_path = opt.dataroot+"/Training/02-21.10way.clean"
                self.train_data = self.get_data(train_data_path)
                # Save the processed model for future loading
                with open(opt.dataroot + 'train_data.pkl', 'wb') as f:
                    dill.dump(self.train_data, f, pickle.HIGHEST_PROTOCOL)
                f.close()

            # Get the processed validation dataset
            if os.path.isfile(opt.dataroot + 'val_data.pkl'):
                print("Loading the processed validation data file")
                with open(opt.dataroot + 'val_data.pkl', 'rb') as in_file:
                    self.val_data = dill.load(in_file)
                in_file.close()
            else:
                self.val_data_path = opt.dataroot + "/Validation/22.auto.clean"
                print("Processing Validation data")
                self.val_data = self.get_data(self.val_data_path)
                with open(opt.dataroot + 'val_data.pkl', 'wb') as f:
                    dill.dump(self.val_data, f, pickle.HIGHEST_PROTOCOL)
                f.close()

            # Get the processed testing dataset
            if os.path.isfile(opt.dataroot + 'test_data.pkl'):
                print("Loading the processed testing data file")
                with open(opt.dataroot + 'test_data.pkl', 'rb') as in_file:
                    self.test_data = dill.load(in_file)
                in_file.close()
            else:
                print("Processing Testing data")
                self.test_data_path = opt.dataroot+"/Testing/23.auto.clean"
                self.test_data = self.get_data(self.test_data_path)
                with open(opt.dataroot + 'test_data.pkl', 'wb') as f:
                    dill.dump(self.test_data, f, pickle.HIGHEST_PROTOCOL)
                f.close()

            if not os.path.isfile(opt.dataroot + 'vocabulary.pkl'):
                with open(opt.dataroot + 'vocabulary.pkl', 'wb') as f:
                    dill.dump(self.vocabulary, f, pickle.HIGHEST_PROTOCOL)
                f.close()
        else:
            self.mode = 'test'
            if os.path.isfile(opt.dataroot + 'test_data.pkl'):
                print("Loading the processed testing data file")
                with open(opt.dataroot + 'test_data.pkl', 'rb') as in_file:
                    self.test_data = dill.load(in_file)
                in_file.close()
            else:
                assert "Missing processed test dataset"

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

        data = defaultdict(dict)
        for idx, line in enumerate(lines):
            source, target, length = self.pre_process_data(line)
            input = []
            for word in source:
                self.vocabulary.add_word(word)
                token = self.vocabulary.word2token[word]
                input.append(token)
            target = [self.vocabulary.word2token[word] for word in target]

            data[idx]['input'] = torch.LongTensor(input)
            data[idx]['target'] = torch.LongTensor(target)
            data[idx]['length'] = length

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
        unwanted = ['``', '\'\'', '&', '!', '?', '@', '#', '$', '%', '\"\"']
        sentence = [word for word in words if word not in unwanted]

        # Add the Start-Of-Sentence (SOS) for the sentence
        # sentence = ['-SOS-'] + sentence + ['-EOS-']
        source = ['-SOS-'] + sentence
        source = source[:self.max_seq_len]
        target = sentence[:self.max_seq_len-1] + ['-EOS-']

        assert len(source) == len(target), "Mis-match in the source and target length"

        length = len(source)
        source.extend(['-PAD-'] * (self.max_seq_len - length))
        target.extend(['-PAD-'] * (self.max_seq_len - length))

        return source, target, length

    def load_data(self, mode, index, batch_size):
        """
        Return the data from a specific index
        """
        if mode == 'train':
            data = self.train_data
        elif mode == 'val':
            data = self.val_data
        else:
            data = self.test_data

        batch = min(batch_size, len(data))

        indices = [i for i in range(index, index+batch_size)]
        source = torch.LongTensor()
        target = torch.LongTensor()
        length = []
        for idx in indices:
            source = torch.cat((source, data[idx]['input'].view(1, -1)), dim=0)
            target = torch.cat((target, data[idx]['target'].view(1, -1)), dim=0)
            length.append(data[idx]['length'])

        return source, target, length

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
