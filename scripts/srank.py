import math
import numpy as np
from jaxtyping import Array, Float, Int, Bool

def covec_of_argsort(argsort: Int[Array, "... n"]) -> Bool[Array, "... math.comb(n,2)"]:
    """
    Operates on the final dimension
    Returns booleans indicating comparisons directions for each pair.
    """
    comparisons = argsort[..., :, None] > argsort[..., None, :]
    return comparisons[..., *np.triu_indices(argsort.shape[-1], k=1)]

rng = np.random.default_rng(0)

n, d = 10, 5
matrix: Float[Array, "n d"] = rng.standard_normal((d, n))

for max_argsort_count in range(1, 100):
    inputs: Float[Array, "m d"] = rng.standard_normal((max_argsort_count, d))
    argsorts: Int[Array, "m n"] = np.unique(np.argsort(inputs @ matrix, axis=-1), axis=0)
    covecs: Int[Array, "m math.comb(n-1,2)"] = covec_of_argsort(argsorts) * 2 - 1
    rank = np.linalg.matrix_rank(covecs)
    assert rank == min(covecs.shape)
