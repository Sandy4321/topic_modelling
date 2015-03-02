import scipy
import logging

from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from loaders import load_documents_to_sparse_matrix


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
