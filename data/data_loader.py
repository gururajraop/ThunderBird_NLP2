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
        if opt.mode == 'train':
            self.mode = 'train'
            self.data_path_eng = opt.dataroot+"/training/hansards.36.2.e"
            self.data_path_fre = opt.dataroot+"/training/hansards.36.2.f"
        else:
            self.data_path_eng = opt.dataroot + "/validation/dev.e"
            self.data_path_fre = opt.dataroot + "/validation/dev.f"

        data_eng = open(self.data_path_eng, 'r', encoding='utf8')
        data_fre = open(self.data_path_fre, 'r', encoding="utf8")

        self.data_eng = data_eng.readlines()
        self.data_fre = data_fre.readlines()

        self.data = self.get_dictionaries(opt.direction)

        self.french_vocab = self.create_french_vocabulary()
        self.eng_vocab = self.create_eng_vocabulary()

    def create_french_vocabulary(self):
        french_vocab = set()
        for line in self.data_fre:
            line = line.strip('\n')
            words = line.split()
            french_vocab.update(words)
        return french_vocab

    def get_french_vocabulary(self):
        return self.french_vocab

    def create_eng_vocabulary(self):
        eng_vocab = set()
        for line in self.data_eng:
            line = line.strip('\n')
            words = line.split()
            eng_vocab.update(words)
        return eng_vocab

    def get_eng_vocabulary(self):
        return self.eng_vocab

    def get_dictionaries(self, direction):

        data_eng = open(self.data_path_eng, 'r', encoding='utf8')
        data_fre = open(self.data_path_fre, 'r', encoding="utf8")

        data_eng = data_eng.readlines()
        data_fre = data_fre.readlines()

        """
        # print(len(train_data_eng), len(train_data_fre))
        eng_to_french_dict = {}  # Key: English sentence, value: List of French translations
        french_to_eng_dict = {}  # Key: French sentence, value: List of English translations
        for eng_fre_pair in zip(train_data_eng, train_data_fre):
            eng_sentence = eng_fre_pair[0].strip()
            fre_sentence = eng_fre_pair[1].strip()
            if eng_sentence in eng_to_french_dict:
                if fre_sentence not in eng_to_french_dict[eng_sentence]:  # Another translation of same english sentence
                    eng_to_french_dict[eng_sentence].append(fre_sentence)
            else:
                eng_to_french_dict[eng_sentence] = [fre_sentence]
            if fre_sentence in french_to_eng_dict:
                if eng_sentence not in french_to_eng_dict[fre_sentence]:  # More than one mapping from french to english
                    french_to_eng_dict[fre_sentence].append(eng_sentence)
            else:
                french_to_eng_dict[fre_sentence] = [eng_sentence]
        """

        if direction == "E2F":
            # return the english to french dictionary
            eng_to_french_dict = [[sentence.strip().split() for sentence in pair] for pair in zip(data_fre, data_eng)]
            return eng_to_french_dict
        else:
            # return the french to english dictionary
            french_to_eng_dict = [[sentence.strip().split() for sentence in pair] for pair in zip(data_eng, data_fre)]
            return french_to_eng_dict

    def load_data(self, key):
        return self.data_dict[key]

    def __len__(self):
        return len(self.size)

