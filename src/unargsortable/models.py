from functools import cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_lm_head(model):
	if model.config.model_type in ["gpt_neox", "gpt_neo"]:
		return model.get_output_embeddings().weight.data.numpy()
	if model.config.model_type == "olmo2":
		return model.lm_head.weight.data.numpy()


def lm_head_of_str(model_name):
	return get_lm_head(get_model(model_name))


@torch.inference_mode()
def inference(model_name, text):
	model = get_model(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokens = tokenizer(text, return_tensors="pt")
	return model(**tokens).logits.numpy()


@cache
def get_model(model_name):
	return AutoModelForCausalLM.from_pretrained(model_name)
