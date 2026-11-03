import numpy as np

def token_embedding(token_ids, embedding_matrix):
    return embedding_matrix[:, token_ids]