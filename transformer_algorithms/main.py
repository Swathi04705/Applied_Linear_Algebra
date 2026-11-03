import numpy as np

from token_embedding import token_embedding
from layer_norm import layer_norm
from attention import attention
from multi_head_attention import multihead_attention


np.random.seed(42)

# 1. Token Embedding
embedding_matrix = np.random.rand(4, 10)
token_ids = np.array([1, 3, 5])

result1 = token_embedding(token_ids, embedding_matrix)

print("TOKEN EMBEDDING")
print(result1)
print("Shape:", result1.shape)


# 2. Layer Normalization
X = np.random.rand(4, 3)
gamma = np.ones(4)
beta = np.zeros(4)

result2 = layer_norm(X, gamma, beta)

print("\nLAYER NORMALIZATION")
print(result2)
print("Shape:", result2.shape)


# 3. Attention
X = np.random.rand(2, 3)
Z = np.random.rand(2, 3)

Wq = np.random.rand(2, 2)
Wk = np.random.rand(2, 2)
Wv = np.random.rand(2, 2)

Mask = np.ones((3, 3), dtype=bool)

result3 = attention(X, Z, Wq, Wk, Wv, Mask)

print("\nATTENTION")
print(result3)
print("Shape:", result3.shape)


# 4. Multi-Head Attention
Wq_heads = [np.random.rand(2, 2) for _ in range(2)]
Wk_heads = [np.random.rand(2, 2) for _ in range(2)]
Wv_heads = [np.random.rand(2, 2) for _ in range(2)]

Wo = np.random.rand(2, 4)

result4 = multihead_attention(
    X, Z,
    Wq_heads,
    Wk_heads,
    Wv_heads,
    Wo,
    Mask
)

print("\nMULTI-HEAD ATTENTION")
print(result4)
print("Shape:", result4.shape)