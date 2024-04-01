import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from fastchat.model import get_conversation_template
from transformers import pipeline as transformer_pipeline
import json
import random

def get_ids(tokenizer, vals, device):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

def my_loss(model, tokenizer, input_str, end_strs, device):

    input_ids = [torch.tensor(tokenizer(f"{input_str}{end_str}").input_ids).to(device) for end_str in end_strs] 
    l = torch.tensor(tokenizer(f"{input_str}").input_ids).to(device).shape[0] - 1
    nested_ids = torch.nested.nested_tensor(input_ids)


    pad_tok = 0
    max_len = max([test_ids1.shape[0] for test_ids1 in input_ids])
    while any([pad_tok in ids for ids in input_ids]):
        pad_tok += 1
    input_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(input_ids), max_len)).to(device)

    attention_mask = torch.ones(input_ids.shape).to(device)
    attention_mask[input_ids == 0] = 0

    labels = input_ids.clone().to(device)
    labels[:, :l] = -100

    res = model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()

# model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b"
# model = GemmaForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#     ).to("cuda:0").eval()

# tokenizer = GemmaTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         use_fast=False
#     )

device = "cuda:0"

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-2b"
model = GemmaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer = GemmaTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         use_fast=False
#     )
pipeline = transformer_pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)


gen_config = model.generation_config
gen_config.max_new_tokens = 64
gen_config.repetition_penalty = 1
gen_config.temperature = 0.7

dataset = json.load(open('dataset.json'))
thesarus = json.load(open('thesaurus.json'))

base_prompt = "What is the best laptop?"
base_prompt_ids = get_ids(tokenizer, base_prompt, device)

target_strs = ["Mac", "MacOS", "Apple"]

loss = my_loss(model, tokenizer, base_prompt, target_strs, device)

print(loss)