import cvxpy as cp, numpy as np

cp.set_num_threads(32)

# Warm start problem: it would be nice to warm start by reusing a problem,
# but each topk is a different problem. We could have v^2 constraints by creating pairwise comparisons,
# but that might make compilation even slower.
class RealizabilitProblem:
	def __init__(self, array):
		raise NotImplementedError
		vocab_size, embed_size = array.shape
		embed = cp.Variable(embed_size)
		scores = np.broadcast_to(array, (vocab_size, vocab_size, embed_size)) @ embed 
		top_scores = scores[topk]
		botk = np.delete(np.arange(vocab_size), topk)
		bottom_scores = scores[botk]
		epsilons = np.array
		constraints = [
			scores >= scores.reshape(vocab_size, 1) + epsilons # Vocab size x vocab size will be way too big
		]
		objective = cp.Maximize(cp.min(top_scores[:-1] - top_scores[1:]))
		self.problem = cp.Problem(objective, constraints)

	def solve(self, topk, warm_start=None):
		raise NotImplementedError
		if warm_start is not None:
			embed.value = warm_start
		self.problem.solve(solver="MOSEK", ignore_dpp=True, canon_backend="SCIPY", **kwargs)
		return self.problem.value >= 1e-5

def realizes(array, topk, warm_start=None, **kwargs):
	vocab_size, embed_size = array.shape
	embed = cp.Variable(embed_size)
	if warm_start is not None:
		embed.value = warm_start
	scores = array @ embed
	top_scores = scores[topk]
	botk = np.delete(np.arange(vocab_size), topk)
	bottom_scores = scores[botk]
	epsilon = 1e-6
	constraints = [
			top_scores[:-1] >= top_scores[1:] + epsilon,
			top_scores[-1] >= bottom_scores + epsilon,
			-10 <= embed,
			embed <= 10
			]
	objective = cp.Maximize(cp.min(top_scores[:-1] - top_scores[1:]))
	problem = cp.Problem(objective, constraints)
	solve_kwargs = dict(solver="MOSEK", ignore_dpp=True, canon_backend="SCIPY") | kwargs
	if solve_kwargs["solver"] == "MOSEK":
		mosek_params = dict(
		MSK_IPAR_FOLDING_USE="MSK_FOLDING_MODE_OFF",
		MSK_IPAR_PRESOLVE_USE="MSK_PRESOLVE_MODE_OFF",
		)
		solve_kwargs["mosek_params"] = mosek_params
	if solve_kwargs["solver"] in ["CPLEX", "GUROBI"]:
		solve_kwargs["reoptimize"] = True
	problem.solve(**solve_kwargs)
	if problem.status == 'optimal':
		sol_scores = array @ embed.value
		sol_topk = np.argsort(-sol_scores)[:len(topk)]
		assert set(sol_topk) == set(topk)
	return problem.value >= 0.1
