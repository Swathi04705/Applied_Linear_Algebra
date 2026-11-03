import numpy as np

def softmax(X, axis=0):
    X = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


def attention(X, Z, Wq, Wk, Wv, Mask):
    Q = Wq @ X
    K = Wk @ Z
    V = Wv @ Z

    S = K.T @ Q
    S = np.where(Mask, S, -np.inf)

    A = softmax(S / np.sqrt(Q.shape[0]), axis=0)

    return V @ A