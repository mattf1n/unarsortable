from bisect import bisect
from functools import partial
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unargsortable.models import get_lm_head
from unargsortable.realizability import realizes
from fire import Fire

def main(model_name="EleutherAI/pythia-70m", solver="MOSEK"):
	rng = np.random.default_rng(0)
	print(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	text = "Some random text"
	tokens = tokenizer(text, return_tensors="pt")
	with torch.inference_mode():
		lm = AutoModelForCausalLM.from_pretrained(model_name)
		logits = lm(**tokens).logits[0, -1, :].numpy()
	argsort = np.argsort(logits)[:1:-1]
	print(np.argmax(logits))
	print(argsort)
	unembed = get_lm_head(lm)
	vocab_size, embed_size = unembed.shape

	max_scale = 16
	scales = np.linspace(0, max_scale, 4096)
	ks = list(range(vocab_size))

	def unargsortable(scale, k): 
		perturbation = rng.normal(scale=scale, size=unembed.shape)
		perturbed_unembed = unembed + unembed
		r = realizes(perturbed_unembed, argsort[:k], solver=solver, verbose=True)
		print(scale, k, r, sep="\t")
		return not r
	
	k = bisect(ks, max_scale, key=lambda k: unargsortable(0, k) + unargsortable(max_scale, k))
	print(scales[bisect(scales, 0.5, key=partial(unargsortable, k=k))])

if __name__ == "__main__":
	Fire(main)
