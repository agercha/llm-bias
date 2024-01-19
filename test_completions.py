import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM)
from fastchat.model import get_conversation_template
# from run_utils import *
import json

def get_replacements(prompt, thesarus):
    prompt_words = prompt.split()
    all_prompts = [prompt_words]
    for ind, word in enumerate(prompt_words):
        if word in thesarus:
            syns = thesarus[word]
            all_prompts = [curr_prompt[:ind] + [syn] + curr_prompt[ind+1:] for syn in syns for curr_prompt in all_prompts]
    return [' '.join(curr_prompt) for curr_prompt in all_prompts]

def get_ids(tokenizer, vals, device):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

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

    return output_ids[0]

def single_successful(gen_str, target_strs):
    gen_str_unpunctuated = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), gen_str))
    gen_str_unpunctuated = gen_str_unpunctuated.upper()
    gen_arr = gen_str_unpunctuated.split()
    present = False
    for prefix in target_strs:
        if prefix.strip().upper() in gen_arr:
            present = True
    return present


def run(local):
    if local:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    else:
        model_path  = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"
        model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda:0").eval()

        tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
        
    if local: device = "cpu"
    else: device = "cuda:0"

    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = model.config.eos_token_id

    dataset = json.load(open('dataset.json'))
    thesarus = json.load(open('thesarus.json'))

    test_size = 100
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 0.5

    for category in dataset:
        if category not in ["browser"]:

            prompt = dataset[category]["top_prompt"]
            brands = dataset[category]["brands"]
            prompts = get_replacements(prompt, thesarus)

            f = open(f"completion_results/{category}.txt", )

            f.write("\t")
            for brand in brands:
                f.write(f"{brand}\t")
                
            for prompt_ind, prompt in enumerate(prompts):
                prompt_ids = get_ids(tokenizer, prompt, device)
                for _ in range(test_size):
                    completion = tokenizer.decode((generate(model, tokenizer, prompt_ids, gen_config=gen_config))).strip()
                    f.write(f"{completion}\t")
                    for brand_ind, brand in enumerate(brands):
                        target_strs = brands[brand]
                        if single_successful(completion, target_strs): 
                            f.write("1\t")
                        else:
                            f.write("0\t")

            f.close()
            
                

run(False)