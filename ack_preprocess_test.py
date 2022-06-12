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
    total_len = window_size * 2 + 1 # previous fixed as 5
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


def ack_extract(fx):
    fx = remove_punctuations(fx)
    fx = remove_stop_words(fx)
    index = locate_pattern(fx)
    valid = True
    extracted_seq = []
    if len(index) == 0:
        print('The results will be deleted at row={} of sheet_name={}'.format(i, 'YES'))
        valid = False
    else:
        for tmp_index in index:
            extracted_seq.append(extract_index(fx, tmp_index))
        # extracted_seq = extract_index(fx, index[-1])
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


def train_svm():
    inst_name = {'TAMU', 'Penn State', 'Florida', 'Purdue', 'UNC', 'Illinios'}
    x_all = []
    y_all = []
    for tmp_inst in inst_name:
        data_file = 'data/' + tmp_inst + '_emb_{}.pickle'.format(window_size)
        with open(data_file, 'rb') as fp:
            data = pickle.load(fp)
        feature = data['x']
        label = data['y']
        x_all.append(feature)
        y_all.append(label)
    y_all = np.concatenate(y_all, axis=0)
    x_all = np.concatenate(x_all, axis=0).reshape(y_all.shape[0], -1)
    print('Feature size for SVM {}'.format(x_all.shape))
    classifier = SVC(C=10)
    classifier.fit(x_all, y_all)
    return classifier


input_data = 'data/Merged_WOS_Ack_Librar_1_31129_03312021.xlsx'
result_data = 'data/Merged_WOS_Ack_Librar_result_window-{}.xlsx'.format(window_size)

inst_name = 'test_sample'

df_tamu_yes = pd.read_excel(input_data)

column_name = df_tamu_yes.columns
idx_fx = column_name.values.tolist().index('FX')
idx_CR = column_name.values.tolist().index('CR')


n_row_yes = len(df_tamu_yes)

svm_classifier = train_svm()
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)


pos_result = []
for i in tqdm(range(n_row_yes)):
    temp_row = df_tamu_yes.loc[i].values.tolist()
    fx = temp_row[idx_fx]
    extracted_seq, valid = ack_extract(fx)
    if valid:
        tmp_label = []
        for sample in extracted_seq:
            x_test = emb_lookup(model, sample)
            x_test = np.reshape(x_test, [1, -1])
            y = svm_classifier.predict(x_test)
            tmp_label.append(y.tolist()[0])
        label = max(tmp_label, key=tmp_label.count)
    else:
        label = -1
    df_tamu_yes.loc[i, 'CR'] = label

pd.DataFrame(df_tamu_yes).to_excel(result_data, index=False, header=True)
