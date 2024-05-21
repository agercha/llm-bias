import os
import json
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
import torch
import matplotlib.pyplot as plt

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
# tokenizer = AutoTokenizer.from_pretrained(
#                 '/Users/annagerchanovsky/Documents/Documents/research/Llama-2-7b-hf',
#                 trust_remote_code=True,
#                 use_fast=False
#             )

entries = os.listdir(f'long_completions_for_user_study/llama3')

lengths = []

for entry in entries:
    completions = json.load(open(f'long_completions_for_user_study/llama3/{entry}'))

    for completion in completions["base_prompt_completions"]:
        completion = completion[17+len(completions["base_prompt"]):]
        lengths.append(len(tokenizer(completion).input_ids))

    for completion in completions["perturbed_prompt_completions"]:
        completion = completion[17+len(completions["perturbed_prompt"]):]
        lengths.append(len(tokenizer(completion).input_ids))


fig = plt.figure(figsize=(14,6))
plt.hist(lengths, bins=128)

# plt.show()
plt.savefig("graphs/llam3lengthhist.png")