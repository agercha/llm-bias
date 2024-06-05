import os
import json
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
import torch

def get_ids(tokenizer, vals, device="cuda:0"):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

def get_len(completion, prompt, tokenizer):
    if 'assistant<|end_header_id|>\n\n' in completion:
        start_ind = completion.index('assistant<|end_header_id|>\n\n')
        completion = completion[start_ind+28:]
    else:
        start_ind = completion.index(prompt)
        completion = completion[start_ind+len(prompt):]
    ids = get_ids(tokenizer, completion)
    return len(ids)

device = "cuda:0"

entries = os.listdir("long_completions_for_user_study/llama3it")
# entries.remove("user_study_json")

f = open("llama3it_lens.txt", "w")

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B-Instruct"
# model = LlamaForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#     ).to("cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
# pipeline = transformer_pipeline(
#     "text-generation",
#     model=model_path,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

for entry in entries:
    res = json.load(open(f"long_completions_for_user_study/llama3it/{entry}")) 

    prompt = res["base_prompt"]

    for completion in res["base_prompt_completions"]:
        l = get_len(completion, prompt, tokenizer)
        f.write(f"{l}\n")

    for completion in res["perturbed_prompt_completions"]:
            l = get_len(completion, prompt, tokenizer)
            f.write(f"{l}\n")
