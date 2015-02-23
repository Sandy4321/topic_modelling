import scipy
import logging

from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

def load_documents_to_sparse_matrix(filename):
    with open(filename, 'r') as collection_file:
        total_docs = int(collection_file.readline())
        total_features = int(collection_file.readline())

        y = scipy.zeros(total_docs)
        X = scipy.zeros((total_docs, total_features))
        row = 0

        for document in collection_file:
            labels, text_features = document.split('\t')[0].split(), document.split('\t')[1].split()
            text_features = zip(text_features[::2], text_features[1::2])
            for label in labels:
                y[row] = label
                for feature in text_features:
                    X[row][int(feature[0])] = float(feature[1])
                row += 1
        X = sp.csr_matrix(X)
    return X, y


def rebuild_answers_for_binary_regression(y, positive_class):
    new_y = deepcopy(y)
    for i in xrange(len(y)):
        new_y[i] = 1 if y[i] == positive_class else 0
    return new_y


def train_one_binary_regression(X, y, positive_class, **kwargs):
    new_y = rebuild_answers_for_binary_regression(y, positive_class)
    lr = LogisticRegression()
    lr.fit(X, new_y)
    return lr


if __name__ == '__main__':
    logging.basicConfig()

    X, y = load_documents_to_sparse_matrix('./data/out.txt')

    for i in xrange(0, 50):
        lr = train_one_binary_regression(X, y, 0)
        y = rebuild_answers_for_binary_regression(y, 0)
        y_pred = lr.predict(X[8000:])
        y_true = y[8000:]
        from sklearn.metrics import f1_score, precision_score, recall_score
        print f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)
