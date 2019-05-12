"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

from nltk.tree import Tree
from collections import defaultdict

class DataLoader():
    """
    The data loading class
    This class handles the data loading and pre-processing of the data
    """

    def __init__(self, opt):
        """
        Initialization fo the DataLoader
        """
        self.input_size = opt.input_size
        self.word2idx = {}
        self.idx2word = []
        if opt.mode == 'train':
            self.mode = 'train'
            self.train_data_path = opt.dataroot+"/Training/02-21.10way.clean"
            self.val_data_path = opt.dataroot + "/Validation/22.auto.clean"
            print("Processing Training data")
            self.train_data, self.train_vocab = self.get_data(self.train_data_path)
            self.train_size = len(self.train_data)
            self.train_vocab_size = len(self.train_vocab)
            print("Training Vocabulary size : ", self.train_vocab_size)
            self.get_word_index_data(self.train_vocab)
            self.tokenized_train_data = self.get_tokenized_data(self.train_data) #Also pads
            print("Processing Validation data")
            self.val_data, self.val_vocab = self.get_data(self.val_data_path)
            self.val_size = len(self.val_data)
            self.val_vocab_size = len(self.val_vocab)
            print("Validation Vocabulary size : ", self.val_vocab_size)
            self.get_word_index_data(self.val_vocab)
            self.tokenized_val_data = self.get_tokenized_data(self.val_data) #Also pads
        else:
            self.mode = 'test'
            self.test_data_path = opt.dataroot+"/Testing/23.auto.clean"
            print("Processing Testing data")
            self.test_data, self.test_vocab = self.get_data(self.test_data_path)
            self.test_size = len(self.test_data)
            self.test_vocab_size = len(self.test_vocab)
            print("Testing Vocabulary size : ", self.test_vocab_size)

    def get_data(self, data_path):
        """
        Obtain the parsed data using the provided data file path

        Parameters:
            data_path  : Path to the data file

        Returns:
            The parsed data in a list format
        """
        file = open(data_path, 'r', encoding='utf8')
        vocabulary = defaultdict(lambda: 0)
        data = []
        lines = file.readlines()
        max_line_len = 0
        for line in lines:
            sentence = self.pre_process_data(line)
            if len(sentence) > max_line_len: max_line_len = len(sentence)
            for word in sentence:
                vocabulary[word] += 1
            data.append(sentence)
        file.close()

        #count is only 1 for the following tokens. Not updated. Not sure if we need the count
        vocabulary['-EOS-'] += 1 #appended when padding the sequence,
        vocabulary['-PAD-'] += 1 #Used for padding
        vocabulary['-UNK-'] += 1 #Used if word is not in the vocabulary.

        print("Maximum Sentence Length = ", max_line_len)

        return data, vocabulary

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

    def get_word_index_data(self, vocabulary):
        for word, count in vocabulary.items():
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx, self.idx2word

    def padded_sequence(self, sequence):
        seq_len = len(sequence)
        padded_sequence = sequence
        if seq_len < self.input_size:
            for word_idx in range(seq_len, self.input_size):
                padded_sequence += ['-PAD-']
            padded_sequence += ['-EOS-']
        if len(padded_sequence) != self.input_size+1:
            print("INPUT SIZE WRONG!!", sequence, len(padded_sequence), self.input_size)

        return padded_sequence

    def get_padded_data(self, data):
        pad_data = []
        for sequence in data:
            pad_data.append(self.padded_sequence(sequence))
        return pad_data

    def get_tokenized_data(self, data):
        tokenized_data = []
        pad_data = self.get_padded_data(data)
        for sentence in pad_data:
            token_sentence = []
            for word in sentence:
                if word not in self.word2idx:
                    token_sentence.append(self.word2idx['-UNK-'])
                else:
                    token_sentence.append(self.word2idx[word])
            tokenized_data.append(token_sentence)
        return tokenized_data

    def load_data(self, index):
        """
        Return the data from a specific index
        """
        if self.mode == 'train':
            return self.train_data[index]
        else:
            return self.test_data[index]

    def __len__(self):
        """
        Return the length of the dataset
        """
        if self.mode == 'train':
            return self.train_size, self.val_size
        else:
            return self.test_size
