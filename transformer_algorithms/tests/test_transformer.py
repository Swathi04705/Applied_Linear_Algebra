
import numpy as np
import pytest

from transformer_algorithms.token_embedding import token_embedding
from transformer_algorithms.attention import softmax, attention
from transformer_algorithms.multi_head_attention import multihead_attention
from transformer_algorithms.layer_norm import layer_norm


# 1. Token Embedding Tests

def test_token_embedding_shape():
    embedding_matrix = np.random.rand(5, 10)
    token_ids = np.array([1, 3, 5])

    result = token_embedding(token_ids, embedding_matrix)

    assert result.shape == (5, 3)


def test_token_embedding_values():
    embedding_matrix = np.arange(20).reshape(4, 5)
    token_ids = np.array([1, 3])

    result = token_embedding(token_ids, embedding_matrix)

    assert np.array_equal(result, embedding_matrix[:, token_ids])


# 2. Softmax Tests

def test_softmax_column_sum():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    result = softmax(X, axis=0)

    assert np.allclose(np.sum(result, axis=0), 1)


def test_softmax_positive_values():
    X = np.array([[1, 2], [3, 4]])

    result = softmax(X, axis=0)

    assert np.all(result >= 0)


# 3. Attention Tests

def test_attention_output_shape():
    X = np.random.rand(4, 3)
    Z = np.random.rand(4, 5)

    Wq = np.random.rand(2, 4)
    Wk = np.random.rand(2, 4)
    Wv = np.random.rand(3, 4)

    Mask = np.zeros((5, 3), dtype=bool)

    result = attention(X, Z, Wq, Wk, Wv, Mask)

    assert result.shape == (3, 3)


def test_attention_finite_output():
    X = np.random.rand(4, 3)
    Z = np.random.rand(4, 5)

    Wq = np.random.rand(2, 4)
    Wk = np.random.rand(2, 4)
    Wv = np.random.rand(3, 4)

    # True means attention is allowed
    Mask = np.ones((5, 3), dtype=bool)

    result = attention(X, Z, Wq, Wk, Wv, Mask)

    assert np.all(np.isfinite(result))

# 4. Multi-Head Attention Tests

def test_multihead_attention_output_shape():
    X = np.random.rand(4, 3)
    Z = np.random.rand(4, 5)

    Wq = [np.random.rand(2, 4), np.random.rand(2, 4)]
    Wk = [np.random.rand(2, 4), np.random.rand(2, 4)]
    Wv = [np.random.rand(3, 4), np.random.rand(3, 4)]

    Wo = np.random.rand(4, 6)

    Mask = np.zeros((5, 3), dtype=bool)

    result = multihead_attention(X, Z, Wq, Wk, Wv, Wo, Mask)

    assert result.shape == (4, 3)


# 5. Layer Normalization Tests

def test_layer_norm_output_shape():
    X = np.random.rand(4, 5)
    gamma = np.ones(4)
    beta = np.zeros(4)

    result = layer_norm(X, gamma, beta)

    assert result.shape == X.shape


def test_layer_norm_mean():
    X = np.random.rand(4, 5)
    gamma = np.ones(4)
    beta = np.zeros(4)

    result = layer_norm(X, gamma, beta)

    assert np.allclose(np.mean(result, axis=0), 0, atol=1e-5)


def test_layer_norm_variance():
    X = np.array([
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [3, 6, 9, 12, 15],
        [4, 8, 12, 16, 20]
    ], dtype=float)

    gamma = np.ones(4)
    beta = np.zeros(4)

    result = layer_norm(X, gamma, beta)

    assert np.allclose(
        np.var(result, axis=0),
        1,
        atol=1e-4
    )