import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle

window_size = 10


def ack_classification(x, y):
    acc_result = []
    n, seq_len, dim = np.shape(x)
    x = np.reshape(x, [n, seq_len * dim])
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(x):
        train_x, train_y = x[train_index], y[train_index]
        test_x, test_y = x[test_index], y[test_index]
        classifier = SVC(C=10)
        classifier.fit(train_x, train_y)
        acc = accuracy_score(test_y, classifier.predict(test_x))
        acc_result.append(acc)
    acc_result = np.array(acc_result)
    acc_mean = acc_result.mean()
    acc_std = acc_result.std()
    return acc_mean, acc_std


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
    acc_mean, acc_std = ack_classification(feature, label)
    print('### Prediction results for {} school acc={} std={}'.format(tmp_inst, acc_mean, acc_std))

x_all = np.concatenate(x_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
acc_mean, acc_std = ack_classification(x_all, y_all)
print('### Prediction results for all school acc={} std={}'.format(acc_mean, acc_std))
# sheet_name = {'TAMU', 'Penn State', 'Florida', 'Purdue', 'UNC', 'Illinios'}
