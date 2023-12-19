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
        torch.tensor(tokenizer(f"{input_str} {control}").input_ids)
        for control in end_strs
    ]

    test_ids = [
        torch.tensor(tokenizer(control).input_ids)
        for control in end_strs
    ]
    pad_tok = 0
    max_len = max([test_ids1.shape[0] for test_ids1 in test_ids])
    while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
        pad_tok += 1
    nested_ids = torch.nested.nested_tensor(test_ids)
    test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))

    if pad_tok >= 0:
        attn_mask = (test_ids != pad_tok).type(test_ids.dtype)
    else:
        attn_mask = None

    res = model.forward(input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=test_ids,
                        return_dict=True)

    print(res)

tokenizer.pad_token_id = 0

# res = tokenizer.decode(generate(model, 
#                                 tokenizer,  
#                                 get_ids(tokenizer, "Hello how are you?")))


my_loss(model, tokenizer, "Hello How are you?", ["Bad", "Ok", "Good"])