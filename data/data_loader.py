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
        self.data_dict = self.get_dictionaries(opt.direction)

    def get_dictionaries(self, direction):
        train_eng = open(self.data_path_eng, 'r', encoding='utf8')
        train_fre = open(self.data_path_fre, 'r', encoding="utf8")
        train_data_eng = train_eng.readlines()
        train_data_fre = train_fre.readlines()

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

        if direction == "E2F":
            # return the english to french dictionary
            return eng_to_french_dict
        else:
            # return the french to english dictionary
            return french_to_eng_dict

    def load_data(self, key):
        return self.data_dict[key]

    def __len__(self):
        return len(self.size)

