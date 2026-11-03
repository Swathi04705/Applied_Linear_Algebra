import numpy as np

def layer_norm(X, gamma, beta, epsilon=1e-5):
    mean = np.mean(X, axis=0, keepdims=True)
    variance = np.var(X, axis=0, keepdims=True)

    X_normalized = (X - mean) / np.sqrt(variance + epsilon)

    return gamma[:, None] * X_normalized + beta[:, None]