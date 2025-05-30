from unargsortable.realizability import realizes
from unargsortable.models import lm_head_of_str, inference
import numpy as np

rng = np.random.default_rng(0)

def realizes_tester(**config):
	vocab_size, embed_size = 1000, 100
	array = rng.standard_normal((vocab_size, embed_size))
	embed = rng.standard_normal(embed_size)
	# argsort is increasing, so use - to get argmax in position 0
	topk = np.argsort(-array @ embed)[:3]
	assert len(topk) == 3
	assert topk[0] == np.argmax(array @ embed) # Just some sanity checks
	small_k = list(range(3))
	large_k = list(range(100))
	kwargs = dict(verbose=True) | config
	assert realizes(array, topk, warm_start=embed, **kwargs)
	assert realizes(array, small_k, **kwargs)
	assert not realizes(array, large_k, **kwargs)

def test_realizes():
	 realizes_tester()

# def test_highs_realizes():
#	 realizes_tester(solver="HIGHS")

def test_gurobi_realizes():
	realizes_tester(solver="GUROBI")

# def test_cplex_realizes():
# 	 realizes_tester(solver="CPLEX")

def test_pythia_realizability():
	 model_name = "EleutherAI/pythia-70m"
	 logits = inference(model_name, "Some text for testing")[0,-1,:]
	 unembed = lm_head_of_str(model_name)
	 topk = np.argsort(-logits)[:100]
	 assert topk[0] == np.argmax(logits)
	 assert realizes(unembed, topk)

# def test_large_realizes():
#	 vocab_size, embed_size = 10_000, 1000
#	 array = rng.standard_normal((vocab_size, embed_size))
#	 large_k = list(range(300))
#	 embed = rng.standard_normal(embed_size)
#	 topk = np.argsort(array @ embed)[:-3:-1]
#	 kwargs = dict(solver="GUROBI", verbose=True)
#	 assert not realizes(array, large_k, **kwargs)
#	 kwargs = dict(solver="MOSEK", verbose=True)
#	 assert not realizes(array, large_k, **kwargs)
#	 assert False
