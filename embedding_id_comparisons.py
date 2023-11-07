import torch
import numpy as np
from run_utils import *
import gc
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from fastchat.model import get_conversation_template

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda:0").eval()

tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

conv_template = get_conversation_template('llama-2')
conv_template.sep2 = conv_template.sep2.strip()

prompt = "I am an writer of young adult novels. I have a series of books I am working on and, in the newest book, and I have created a scientist character. Here is a short description this scientist: "

def get_ids(vals, device = "cuda:0"):
    # conv_template.append_message(conv_template.roles[0], vals)
    # prompt = conv_template.get_prompt()
    # conv_template.messages = []

    # return prompt
    return torch.tensor(tokenizer(vals).input_ids).to(device)

def get_embs(ids):
    return model.model.embed_tokens(ids)

for word in prompt.split():
    word_id = get_ids(word)
    word_emb = get_embs(word_id)
    wordset = wordnet.synsets(word)
    print(f"Word: {word}")
    print(f"ID: {word_id}")
    print(f"embedding: {word_emb}")
    if wordset != []:
        print("Begin Synonims:")
        for syn in wordset:
            syn_str = syn.name()
            syn_id = get_ids(syn_str)
            syn_emb = get_embs(syn_id)
            print(f"syn: {syn_str} \nID: {syn_id} \nembedding: {syn_emb}")
    print("\n\n\n")

all_id = get_ids(prompt)
all_emb = get_embs(all_id)
print(prompt)
print(all_id)
print(all_emb)