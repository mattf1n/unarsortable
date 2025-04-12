import vllm
# from datasets import load_dataset
# from transformers import AutoTokenizer

model_name = "allenai/OLMo-2-1124-7B"
penultimate_checkpoint = "stage1-step928000-tokens3893B"
final_checkpoint = "stage1-step928646-tokens3896B"

penultimate_llm = vllm.LLM(model=model_name, revision=penultimate_checkpoint, max_logprobs=1000)

final_llm = vllm.LLM(model=model_name, revision=penultimate_llm, max_logprobs=1000)

gen = penultimate_llm.generate("hi")

print(gen)

# dataset = load_dataset("allenai/dolma", streaming=True, split="train").shuffle()
# print(next(iter(dataset)))


