from math import comb
import numpy as np
from jaxtyping import Array, Float, Int, Bool

def covec_of_argsort(argsort: Int[Array, "... n"]) -> Bool[Array, "... comb(n,2)"]:
    """
    Operates on the final dimension
    Returns booleans indicating comparisons directions for each pair.
    """
    comparisons = argsort[..., :, None] > argsort[..., None, :]
    dedup_comparisons = comparisons[..., *np.triu_indices(argsort.shape[-1], k=1)]
    return dedup_comparisons

rng = np.random.default_rng(0)

n, d = 10, 5
matrix: Float[Array, "n d"] = rng.standard_normal((d, n))

# print("Samples", "Rank", sep="\t")
for max_argsort_count in range(comb(n, 2), comb(n, 2) + 1000):
    inputs: Float[Array, "m d"] = rng.standard_normal((max_argsort_count, d))
    argsorts: Int[Array, "m n"] = np.unique(np.argsort(inputs @ matrix, axis=-1), axis=0)
    covecs: Int[Array, "m comb(n,2)"] = covec_of_argsort(argsorts) * 2 - 1
    U, S, V = np.linalg.svd(covecs, full_matrices=False)
    np.testing.assert_allclose(np.sign(U @ np.diag(S) @ V), covecs)
    k = 1
    while not (np.sign(U[:, :k] @ np.diag(S[:k]) @ V[:k, :]) == covecs).all() and k < min(covecs.shape):
        k += 1
    print(len(argsorts), k, sep="\t")
