"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

from nltk.tree import Tree

class DataLoader():
    """
    The data loading class
    This class handles the data loading and pre-processing of the data
    """

    def __init__(self, opt):
        """
        Initialization fo the DataLoader
        """
        self.size = 0
        if opt.mode == 'train':
            self.mode = 'train'
            self.train_data_path = opt.dataroot+"/Training/02-21.10way.clean"
            self.val_data_path = opt.dataroot + "/Validation/22.auto.clean"
            print("Processing Training data")
            self.train_data = self.get_data(self.train_data_path)
            print("Processing Validation data")
            self.val_data = self.get_data(self.val_data_path)
        else:
            self.mode = 'test'
            self.test_data_path = opt.dataroot+"/Testing/23.auto.clean"
            print("Processing Testing data")
            self.test_data = self.get_data(self.test_data_path)

    def get_data(self, data_path):
        """
        Obtain the parsed data using the provided data file path

        Parameters:
            data_path  : Path to the data file

        Returns:
            The parsed data in a list format
        """
        file = open(data_path, 'r', encoding='utf8')
        data = []
        lines = file.readlines()
        for line in lines:
            sentence = self.pre_process_data(line)
            data.append(sentence)

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
        return sentence

    def load_data(self, key):
        pass

    def __len__(self):
        pass

