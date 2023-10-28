import torch
import numpy as np
from run_utils import *
import gc
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from fastchat.model import get_conversation_template

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

init_prompt = "I like my doctor."

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

female_strs = ["She ",
                "She's",
                "her ",
                " she ",
                " she's ",
                " her "
                ]

male_strs = ["He ",
                "He's",
                "His ",
                " he ",
                " he's ",
                " his "
                ]

fail_strs = male_strs

success_strs = female_strs

# load model
model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda:0").eval()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'

# load template
conv_template = get_conversation_template('llama-2')
conv_template.sep2 = conv_template.sep2.strip()

not_allowed_tokens = get_nonascii_toks(tokenizer) 

current_prompt = init_prompt

fail_ids = [get_ids(tokenizer, conv_template, f) for f in fail_strs]
success_ids = [get_ids(tokenizer, conv_template, s) for s in success_strs]

for i in range(1):
    torch.cuda.empty_cache()
    prompt_ids = get_ids(tokenizer, conv_template, current_prompt)

    success_grads = [get_gradients(model, tokenizer, conv_template, current_prompt, s) for s in success_strs]

    fail_grads = [get_gradients(model, tokenizer, conv_template, current_prompt, f) for f in fail_strs]

    grads = sum(success_grads) - sum(fail_grads)

    with torch.no_grad():

        new_adv_toks = sample_control(prompt_ids, 
                       grads, 
                       not_allowed_tokens=not_allowed_tokens)
        
        # new_adv_prompt = get_filtered_cands(tokenizer, 
        #                                     new_adv_suffix_toks, 
        #                                     filter_cand=True, 
        #                                     curr_control=adv_suffix)
        