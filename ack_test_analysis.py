import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.svm import SVC
import gensim
from tqdm import tqdm

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


def remove_stop_words(fx):
    fx_new = []
    for i, word_ in enumerate(fx):
        if word_ not in stop_words:
            if re.search(r"\W", word_):
                pass
            else:
                fx_new.append(word_)
    return fx_new


def extract_index(fx, index):
    seq_ext = fx[index-window_size:index+window_size + 1]
    if len(seq_ext) == 7:
        pass
    else:
        add_len = 7 - len(seq_ext)
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


def train_svm():
    inst_name = {'TAMU', 'Penn State', 'Florida', 'Purdue', 'UNC', 'Illinios'}
    x_all = []
    y_all = []
    for tmp_inst in inst_name:
        data_file = 'data/' + tmp_inst + '_emb.pickle'
        with open(data_file, 'rb') as fp:
            data = pickle.load(fp)
        feature = data['x']
        label = data['y']
        x_all.append(feature)
        y_all.append(label)
    y_all = np.concatenate(y_all, axis=0)
    x_all = np.concatenate(x_all, axis=0).reshape(y_all.shape[0], -1)
    classifier = SVC(C=10)
    classifier.fit(x_all, y_all)
    return classifier


result_data = 'data/Merged_WOS_Ack_Librar_result_window-{}.xlsx'.format(window_size)
inst_name = 'test_sample'
print('Open {} dataset'.format(result_data))

df_tamu_yes = pd.read_excel(result_data)

aa = df_tamu_yes.loc[:, 'CR'].value_counts()
print(aa)

column_name = df_tamu_yes.columns
idx_fx = column_name.values.tolist().index('FX')
idx_CR = column_name.values.tolist().index('CR')


n_row_yes = len(df_tamu_yes)

