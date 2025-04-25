# from datasets import load_dataset
from bisect import bisect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unargsortable.realizability import realizes
from unargsortable.models import get_lm_head

@torch.inference_mode()
def main():
	model = "pythia-70m"
	with open("config/models.toml") as file:
		model_config = tomllib.load(file)[model]
	model_name = model_config["name"]
	checkpoint1 = model_config["ckpt1"]
	checkpoint2 = model_config["ckpt2"]

	print("Loading LLM1")
	llm1 = AutoModelForCausalLM.from_pretrained(model_name, revision=checkpoint1)

	print("Loading LLM2")
	llm2 = AutoModelForCausalLM.from_pretrained(model_name, revision=checkpoint2)
	unembed2 = get_lm_head(model2)

	print("Loading tokenizer")
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
	tokens = tokenizer(text, return_tensors="pt")
	token_count = len(tokens.input_ids[0])

	print("Running inference on LLM1")
	# dataset = load_dataset("allenai/dolma", streaming=True, split="train").shuffle()
	# print(next(iter(dataset)))
	output1 = llm1(**tokens, output_hidden_states=True)
	logits1 = output1.logits[0]
	hidden_states1 = output1.hidden_states[-1][0].numpy()

	print("Testing realizability")
	outfile = "data/pythia_ins_pt.dat"
	for i, logits in enumerate(logits1):
		def key(k):
			topk = torch.topk(logits, k).indices.numpy()
			realizable = realizes(unembed2, topk, verbose=True) 
			print(k, realizable)
			return not realizable
		k = bisect(list(range(512)), 0.5, lo=384, hi=448, key=key)
		with open(outfile, "a") as file:
			print(i, k, file=file)




if __name__ == "__main__":
	main()
