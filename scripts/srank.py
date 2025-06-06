import sys
from bisect import bisect
from typing import Literal
from functools import partial
from math import comb
import numpy as np
import jax, jax.numpy as jnp
import jaxopt
from fire import Fire
from tqdm import tqdm

type Array = np.ndarray

rng = np.random.default_rng(0)


def covec_of_argsort(argsort: Array) -> Array:
    """
    Operates on the final dimension
    Returns booleans indicating comparisons directions for each pair.
    """
    comparisons = argsort[..., :, None] >= argsort[..., None, :]
    dedup_comparisons = comparisons[..., *np.triu_indices(argsort.shape[-1], k=1)]
    return dedup_comparisons


def logistic_loss(params, B):
    L, R = params
    p = jax.nn.sigmoid(L @ R)
    return -jnp.sum(B * jnp.log(p) + (1 - B) * jnp.log(1 - p))


def logistic_priciple_component_analysis(B: Array, k: int) -> bool:
    L = rng.standard_normal((B.shape[0], k)) / jnp.sqrt(k)
    R = rng.standard_normal((k, B.shape[1])) / jnp.sqrt(k)
    solver = jaxopt.LBFGS(fun=logistic_loss)
    (L, R), state = solver.run((L, R), B=B)
    return bool((jnp.round(jax.nn.sigmoid(L @ R)).astype(int) == B).all())


def LP(B, k): 
    "This one seems harder. How do we get A?"


def SVD(U, S, V, B: Array, k: int) -> bool:
    pred = np.sign(U[:, :k] @ np.diag(S[:k]) @ V[:k, :]).astype(int)
    return (pred == (B * 2 - 1)).all()


def main(
    n: int,
    d: int,
    m: int,
    method: Literal["svd", "lpca"],
    start: int = 0,
    start_type: Literal["rel", "abs"] = "rel",
    skip: int = 1,
    interpolation: Literal["linear", "log"] = "linear",
    num: int | None = None
):
    matrix: Array = rng.standard_normal((d, n))
    if start_type == "rel":
        start = comb(n, 2) + start
    stop = comb(n, 2) + m
    if interpolation == "linear":
        sample_counts = range(start, stop, skip)
    elif interpolation == "log":
        if num is None:
            raise Exception("num must not be none if interpolation is log")
        sample_counts = np.logspace(np.log10(start), np.log10(stop), num=num).astype(int)
    for max_argsort_count in tqdm(sample_counts):
        inputs: Array = rng.standard_normal((max_argsort_count, d))
        argsorts: Array = np.unique(
            np.argsort(inputs @ matrix, axis=-1), axis=0
        )
        # print(argsorts.shape, file=sys.stderr)
        covecs: Array = covec_of_argsort(argsorts)
        if method == "svd":
            U, S, V = np.linalg.svd(covecs * 2 - 1, full_matrices=False)
            np.testing.assert_allclose(np.sign(U @ np.diag(S) @ V), covecs * 2 - 1)
            is_rank_upper_bound = partial(SVD, U, S, V)
        elif method == "lpca":
            is_rank_upper_bound = logistic_priciple_component_analysis
        key = partial(is_rank_upper_bound, covecs.astype(int))
        bound = bisect(list(range(comb(n, 2))), 0.5, key=key)
        print(len(argsorts), bound, sep="\t", flush=True)


if __name__ == "__main__":
    Fire(main)
