from bisect import bisect
import sys
from tqdm import tqdm
import numpy as np
import cvxpy as cp
import scipy

v, d = 10_000, 100
trial_count = 30
rng = np.random.default_rng()

def expected_overlap(k, trial_count):
    successes = 0
    embed = cp.Variable(d)
    unembed = cp.Parameter((v, d))
    scores = unembed @ embed
    top_scores = scores[:k]
    bottom_scores = scores[k:]
    constraints = [
            top_scores[:-1] <= top_scores[1:],
            top_scores[-1] <= bottom_scores,
            ]
    objective = cp.Maximize(cp.min(top_scores[1:] - top_scores[:-1]))
    problem = cp.Problem(objective, constraints)
    for trial in tqdm(range(trial_count)):
	array = rng.standard_normal((v, d))
        unembed.value = array
	print(realizes(unembed, np.arange(k)))
        problem.solve(solver="MOSEK", ignore_dpp=True)
        successes += problem.value >= 1e-5
    print(k, successes, trial_count, file=sys.stderr)
    return scipy.stats.binomtest(k=successes, n=trial_count)

center = bisect(list(range(2, d)), 0, key=lambda k: (expected_overlap(k, 2).statistic < 0.5))

# center = 240
print(center, file=sys.stderr)

for k in (2, *range(max(3, center - 10), min(d-1, center+20)), d):
    result = expected_overlap(k, trial_count)
    ci = result.proportion_ci()
    print(k, result.statistic, result.pvalue, ci.low, ci.high, sep="\t", flush=True)
