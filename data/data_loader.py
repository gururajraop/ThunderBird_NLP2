"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""



class DataLoader():
    """

    """

    def __init__(self, opt):
        """

        :param opt:
        """
        self.size = 0
        self.data_dict = self.get_dictionaries(opt.direction)

    def get_dictionaries(self, direction):
        if direction == "E2F":
            # return the english to french dictionary
            return
        else:
            # return the french to english dictionary
            return

    def load_data(self):
        pass

    def __len__(self):
        return len(self.size)

