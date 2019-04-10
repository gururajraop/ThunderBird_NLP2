train_e_path = 'training/hansards.36.2.e'
train_f_path = 'training/hansards.36.2.f'

train_eng = open(train_e_path, 'r', encoding='utf8')
train_fre = open(train_f_path, 'r',encoding="utf8")
train_data_eng = train_eng.readlines()
train_data_fre = train_fre.readlines()

print(len(train_data_eng), len(train_data_fre))
eng_to_french_dict = {} # Key: English sentence, value: List of French translations
french_to_eng_dict = {} # Key: French sentence, value: List of English translations
for eng_fre_pair in zip(train_data_eng, train_data_fre):
    eng_sentence = eng_fre_pair[0]
    fre_sentence = eng_fre_pair[1]
    if eng_sentence in eng_to_french_dict:
        if eng_to_french_dict[eng_sentence] != fre_sentence: #Another translation of same english sentence
            eng_to_french_dict[eng_sentence].append(fre_sentence)
    else:
        eng_to_french_dict[eng_sentence] = [fre_sentence]
    if fre_sentence in french_to_eng_dict:
        if french_to_eng_dict[fre_sentence] != eng_sentence: #More than one mapping from french to english
            french_to_eng_dict[fre_sentence].append(eng_sentence)
    else:
        french_to_eng_dict[fre_sentence] = [eng_sentence]


