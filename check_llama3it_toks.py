import os
import json
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
import torch

def get_ids(tokenizer, vals, device="cuda:0"):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

def get_len(completion, prompt, tokenizer, modelname):
    if "llama3" == modelname:
        completion = completion[17+len(prompt):]
    elif "gemma" in modelname:
        completion = completion[57+len(prompt):]
    elif 'assistant<|end_header_id|>\n\n' in completion:
        start_ind = completion.index('assistant<|end_header_id|>\n\n')
        completion = completion[start_ind+28:]
    else:
        start_ind = completion.index(prompt)
        completion = completion[start_ind+len(prompt):]
    ids = get_ids(tokenizer, completion)
    return len(ids)

# ****************************************************************************

modelname = "llama3it"

device = "cuda:0"

entries = os.listdir("long_completions_for_user_study/llama3it")

f = open("llama3it_lens.txt", "w")

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

for entry in entries:
    print(modelname, entry)
    res = json.load(open(f"long_completions_for_user_study/llama3it/{entry}")) 

    prompt = res["base_prompt"]
    for completion in res["base_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer)
        f.write(f"{l}\n")

    prompt = res["perturbed_prompt"]
    for completion in res["perturbed_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer)
        f.write(f"{l}\n")

# ****************************************************************************

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

modelname = "llama3"

entries = os.listdir("long_completions_for_user_study/llama3")
for entry in entries:
    if entry.split(".")[0][-3:] != "__0": entries.remove(entry)

f = open("llama3_lens.txt", "w")

for entry in entries:
    print(modelname, entry)
    res = json.load(open(f"long_completions_for_user_study/llama3/{entry}")) 

    prompt = res["base_prompt"]
    for completion in res["base_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer)
        f.write(f"{l}\n")

    prompt = res["perturbed_prompt"]
    for completion in res["perturbed_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer, modelname)
        f.write(f"{l}\n")

# ****************************************************************************

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)

modelname = "gemma7bit"

entries = os.listdir("long_completions_for_user_study/gemma7bit")
    
f = open("gemma_lens.txt", "w")

for entry in entries:
    print(modelname, entry)
    res = json.load(open(f"long_completions_for_user_study/gemma7bit/{entry}")) 

    prompt = res["base_prompt"]
    for completion in res["base_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer)
        f.write(f"{l}\n")

    prompt = res["perturbed_prompt"]
    for completion in res["perturbed_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer, modelname)
        f.write(f"{l}\n")