import numpy as np
import tensorflow as tf


def sparse_matrix_to_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(
            indices,
            np.array(coo.data, dtype=np.float32),
            coo.shape)


def sparse_dropout(X, dropout_rate):
    X_drop_val = tf.nn.dropout(X.values, dropout_rate)
    return tf.SparseTensor(X.indices, X_drop_val, X.dense_shape)
