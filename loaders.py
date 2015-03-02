# encoding=utf-8
#author: Bocharov Ivan
import scipy


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
