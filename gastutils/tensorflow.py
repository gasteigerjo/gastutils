import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def sparse_matrix_to_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(
            indices,
            np.array(coo.data, dtype=np.float32),
            coo.shape)


def matrix_to_tensor(X):
    if sp.issparse(X):
        return sparse_matrix_to_tensor(X)
    else:
        return tf.constant(X, dtype=tf.float32)


def sparse_dropout(X, dropout_rate):
    X_drop_val = tf.nn.dropout(X.values, dropout_rate)
    return tf.SparseTensor(X.indices, X_drop_val, X.dense_shape)


def mixed_dropout(X, keep_prob):
    if isinstance(X, tf.SparseTensor):
        return sparse_dropout(X, keep_prob)
    else:
        return tf.nn.dropout(X, keep_prob)
