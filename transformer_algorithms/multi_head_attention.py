import numpy as np
from attention import attention


def multihead_attention(X, Z, Wq, Wk, Wv, Wo, Mask):
    heads = []

    num_heads = len(Wq)

    for i in range(num_heads):
        head_output = attention(
            X, Z,
            Wq[i],
            Wk[i],
            Wv[i],
            Mask
        )

        heads.append(head_output)

    concatenated = np.vstack(heads)

    return Wo @ concatenated