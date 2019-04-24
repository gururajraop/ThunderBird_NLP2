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
            self.val_path_eng = opt.dataroot + "/validation/dev.e"
            self.val_path_fre = opt.dataroot + "/validation/dev.f"

            self.val_data = self.get_dictionaries(self.val_path_eng, self.val_path_fre, opt.direction)
        else:
            self.mode = 'test'
            self.data_path_eng = opt.dataroot+"/testing/test/test.e"
            self.data_path_fre = opt.dataroot+"/testing/test/test.f"

        self.data = self.get_dictionaries(self.data_path_eng, self.data_path_fre, opt.direction)

        data_eng = open(self.data_path_eng, 'r', encoding='utf8')
        data_fre = open(self.data_path_fre, 'r', encoding="utf8")

        self.data_eng = data_eng.readlines()
        self.data_fre = data_fre.readlines()

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

    def get_dictionaries(self, data_path_eng, data_path_fre, direction):

        data_eng = open(data_path_eng, 'r', encoding='utf8')
        data_fre = open(data_path_fre, 'r', encoding="utf8")

        data_eng = data_eng.readlines()
        data_fre = data_fre.readlines()

        if direction == "E2F":
            # return the english to french dictionary
            eng_to_french_dict = []
            for fre, eng in zip(data_fre, data_eng):
                fre_sentence = fre.strip().split()
                eng_sentence = eng.strip().split()
                eng_sentence.append("NULL")
                eng_to_french_dict.append([eng_sentence, fre_sentence])

            return eng_to_french_dict

        else:
            # return the french to english dictionary
            french_to_eng_dict = []
            for fre, eng in zip(data_fre, data_eng):
                fre_sentence = fre.strip().split()
                eng_sentence = eng.strip().split()
                fre_sentence.append("NULL")
                french_to_eng_dict.append([fre_sentence, eng_sentence])

            return french_to_eng_dict

    def load_data(self, key):
        return self.data_dict[key]

    def __len__(self):
        return len(self.size)

