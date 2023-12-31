import torch
import numpy as np
from run_utils import *
import gc
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from fastchat.model import get_conversation_template
import datetime

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


def generate(model, tokenizer, input_ids, gen_config=None):

    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32
        
    input_ids = input_ids.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)
    
    print(output_ids)
    print(dir(output_ids))

    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels = labels, output_hidden_states = True)

    return output_ids[0]

def my_loss(model, tokenizer, input_str, end_strs):
    input_ids = [
        torch.tensor(tokenizer(f"{input_str} {control}").input_ids).to("cuda:0")
        for control in end_strs
    ]
    l = torch.tensor(tokenizer(f"{input_str} ").input_ids).to("cuda:0").shape[0]
    pad_tok = 0
    max_len = max([test_ids1.shape[0] for test_ids1 in input_ids])
    while any([pad_tok in ids for ids in input_ids]):
        pad_tok += 1
    nested_ids = torch.nested.nested_tensor(input_ids)
    input_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(input_ids), max_len)).to("cuda:0")


    labels = input_ids.clone() 
    labels[:,:l] = -100

    res = model.forward(input_ids=input_ids,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()

tokenizer.pad_token_id = 0

# res = tokenizer.decode(generate(model, 
#                                 tokenizer,  
#                                 get_ids(tokenizer, "Hello how are you?")))

all_prompts = ["Hello How are you?", "Hey what's up hows it going"]

l1 = torch.FloatTensor([my_loss(model, tokenizer, p, ["Terrible", "Bad", "Awful"]) for p in all_prompts])
l2 = torch.FloatTensor([my_loss(model, tokenizer, p, ["Good", "Awesome", "Amazing"]) for p in all_prompts])

print(l1, l2)
print(l1.argmin())
print(l2.argmin())