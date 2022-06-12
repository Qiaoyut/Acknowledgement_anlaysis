import numpy as np
import pandas as pd
import re
import pickle as plk
from nltk.corpus import stopwords
# nltk.download('stopwords')
import gensim

window_size = 10
stop_words = set(stopwords.words('english'))
ack_patterns = {'library': 'library', 'libraries': 'libraries', 'librarian': 'librarian', 'librarians': 'librarians', 'librarys': 'librarys', 'librarianship': 'librarianship'}


def locate_pattern(str_list_origin):
    pattern = '+++++++++++++++'
    lib_patterns = ack_patterns
    result = []
    for i, sub_str in enumerate(str_list_origin):
        if str_list_origin[i] in lib_patterns:
            result.append(i)
        else:
            pass
    return result


def remove_stop_words(fx, fx_library):
    fx_new = []
    fx_library_new = []
    for i, word_ in enumerate(fx):
        if word_ not in stop_words:
            if re.search(r"\W", word_):
                pass
            else:
                fx_new.append(word_)
                fx_library_new.append(fx_library[i])
    return fx_new, fx_library_new


def extract_index(fx, index):
    seq_ext = fx[index-window_size:index+window_size + 1]
    total_len = window_size * 2 + 1
    if len(seq_ext) == total_len:
        pass
    else:
        add_len = total_len - len(seq_ext)
        seq_ext += ['word'] * add_len
    return seq_ext


def remove_punctuations(fx):
    fx = fx.lower()
    fx = fx.replace('(', ' ')
    fx = fx.replace(')', ' ')
    fx = fx.replace('[', ' ')
    fx = fx.replace(']', ' ')
    fx = fx.replace('\'', ' ')
    fx = fx.replace('/', ' ')
    fx = fx.replace('&', ' ')
    fx = fx.replace(',', ' ')
    fx = fx.replace('.', ' ')
    fx = fx.replace('?', ' ')
    fx = fx.replace('<', ' ')
    fx = fx.replace('>', ' ')
    fx = fx.replace('\\', ' ')
    fx = fx.replace('!', ' ')
    fx = fx.replace('"', ' ')
    fx = fx.replace(':', ' ')
    fx = fx.replace(';', ' ')
    fx = fx.replace('-', ' ')
    fx = fx.split(' ')
    fx = list(filter(None, fx))
    return fx


def ack_extract(fx, fx_library, row):

    fx = remove_punctuations(fx)
    fx_library = remove_punctuations(fx_library)

    fx, fx_library = remove_stop_words(fx, fx_library)
    index = locate_pattern(fx)
    valid = True
    extracted_seq = []
    if len(index) == 0:
        print('The results will be deleted at row={} of sheet_name={}'.format(i, 'YES'))
        valid = False
    else:
        extracted_seq = extract_index(fx, index[-1])
        if len(index) > 1:
            print('There are more than one key words at row={} of sheet_name={}'.format(i, ' YES'))

    return extracted_seq, valid


def emb_lookup(model, seq_text):
    result = []
    for word in seq_text:
        try:
            word_emb = model[word]
        except :
            word_emb = model['word']
        result.append(word_emb)
    result = np.stack(result, axis=0)
    return result


input_data = 'data/Librar-exeample-WoS.xlsx'
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)

for inst_name in {'TAMU', 'Penn State', 'Florida', 'Purdue', 'UNC', 'Illinios'}:

    save_file = 'data/' + inst_name + '_emb_{}.pickle'.format(window_size)
    # sheet_name = {'TAMU', 'Penn State', 'Florida', 'Purdue', 'UNC', 'Illinios'}

    df_tamu_yes = pd.read_excel(input_data, sheet_name='{} - YES'.format(inst_name))
    df_tamu_no = pd.read_excel(input_data, sheet_name='{} - NO'.format(inst_name))
    column_name = df_tamu_yes.columns

    df_yes = df_tamu_yes.iloc[:, [0, 2, 3]]
    df_no = df_tamu_no.iloc[:, [0, 2, 3]]

    n_row_yes = len(df_yes)
    n_row_no = len(df_tamu_no)

    pos_result = []
    for i in range(n_row_yes):
        temp_row = df_yes.loc[i].values.tolist()
        extracted_seq, valid = ack_extract(temp_row[1], temp_row[2], i)
        if valid:
            pos_result.append(extracted_seq)
        else:
            print('---- Skip the {} sample'.format(i))

    print('Extracted {} pos_samples!'.format(len(pos_result)))

    neg_result = []
    for i in range(n_row_no):
        temp_row = df_no.loc[i].values.tolist()
        extracted_seq, valid = ack_extract(temp_row[1], temp_row[2], i)
        if valid:
            neg_result.append(extracted_seq)
        else:
            print('---- Skip the {} sample'.format(i))

    print('Extracted {} neg_samples!'.format(len(neg_result)))
    #

    print('--- Start embedding for {} ack text'.format(inst_name))
    x = []
    y = []
    for i, sample in enumerate(pos_result):
        tmp_emb = emb_lookup(model, sample)
        x.append(tmp_emb)
        y.append(float(1))

    for i, sample in enumerate(neg_result):
        tmp_emb = emb_lookup(model, sample)
        x.append(tmp_emb)
        y.append(float(0))

    x = np.stack(x, axis=0)
    y = np.array(y)
    with open(save_file, 'wb') as fp:
        plk.dump({'x': x, 'y': y}, fp, protocol=plk.HIGHEST_PROTOCOL)
    # np.save(save_file, {'x': x, 'y': y})
    print('#### Save {} file for {} pos and {} neg samples'.format(save_file, len(pos_result), len(neg_result)))


