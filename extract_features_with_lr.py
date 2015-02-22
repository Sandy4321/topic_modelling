import scipy

from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression


def load_documents_to_sparse_matrix(filename):
    with open(filename, 'r') as collection_file:
        total_docs = int(collection_file.readline())
        total_features = int(collection_file.readline())

        y = scipy.zeros(total_docs)
        X = sp.csr_matrix(shape=(total_docs, total_features), dtype='float32')
        row = 0

        for document in collection_file:
            labels, text_features = document.split('\t')[0], document.split('\t')[1].split()
            for label in labels:
                y[row] = label
            for i in xrange(0, len(text_features)/2):
                X[row][int(text_features[i])] = text_features[i+1]
            row += 1
    return X, y


print load_documents_to_sparse_matrix()
