import scipy

from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression


def load_documents_to_sparse_matrix(filename):
    with open(filename, 'r') as collection_file:
        total_docs = int(collection_file.readline())
        total_features = int(collection_file.readline())
        print total_docs, total_features

        y = scipy.zeros(total_docs)
        X = scipy.zeros((total_docs, total_features))
        print X.shape
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


X, y = load_documents_to_sparse_matrix('./data/out.txt')

lr = LogisticRegression()
lr.fit(X[:8000], y[:8000])
y_pred = lr.predict(X[8000:])
y_true = y[8000:]

print lr.predict_proba(X)

from sklearn.metrics import f1_score, precision_score, recall_score
print f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)
