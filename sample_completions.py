import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from fastchat.model import get_conversation_template
import json
import random
import copy
import os

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
    # gen_arr = gen_str_unpunctuated.split()
    present = False
    for prefix in target_strs:
        if prefix.strip().upper() in gen_str_unpunctuated:
            present = True
    return present

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

def run(modelname):
    if modelname == "gpt":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    elif modelname == "llama":
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
    elif modelname == "gemma2b":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-2b"
        model = GemmaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda:0").eval()

        tokenizer = GemmaTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
    elif modelname == "gemma7b":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b"
        model = GemmaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda:0").eval()

        tokenizer = GemmaTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
    elif modelname == "gemma7bit":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b"
        model = GemmaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda:0").eval()

        tokenizer = GemmaTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
        
    if modelname == "gpt": device = "cpu"
    else: device = "cuda:0"

    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = model.config.eos_token_id

    dataset = json.load(open('dataset.json'))
    thesarus = json.load(open('thesaurus.json'))

    # completions_json_file = json.load(open('completions_temp_1_0.json'))
    # old_completions_json_file = json.load(open('base_completions_temp_1_0.json'))

    test_size = 1000
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 1.00
    
    if "llama" not in modelname:   
        gen_config.temperature = 0.7 
    
    while True:
        category = random.choice(list(dataset.keys()))
        brand = random.choice(list(dataset[category]["brands"].keys()))
        base_prompt = random.choice(list(dataset[category]["prompts"]))
        base_prompt_ids = get_ids(tokenizer, base_prompt, device)
        base_prompt_original_ind = dataset[category]["prompts"].index(base_prompt)

        completed_files = os.listdir("adversarial_completions_llama_temp1")

        original_json = json.load(open(f'base_completions_llama_temp1/{category}.json'))

        if f"{category}_{base_prompt_original_ind}.json" not in completed_files and base_prompt_original_ind in original_json:

            base_completions = original_json[base_prompt_original_ind]["base_prompt_completions"]

            target_strs = dataset[category]["brands"][brand]

            base_completions_success = 0

            for base_completion in base_completions:
                if single_successful(base_completion, target_strs):
                    base_completions_success += 1

            if base_completions_success / len(base_completions) > 0.1:

                perturbed_prompts = get_replacements(base_prompt, thesarus)

                base_prompt_ind = perturbed_prompts.index(base_prompt.strip())

                losses = torch.zeros(len(perturbed_prompts))

                for prompt_ind, curr_prompt in enumerate(perturbed_prompts):
                    losses[prompt_ind] = my_loss(model, tokenizer, curr_prompt, target_strs, device)

                perturbed_prompt_ind = torch.argmin(losses).item()
                perturbed_prompt = perturbed_prompts[perturbed_prompt_ind]
                perturbed_prompt_ids = get_ids(tokenizer, perturbed_prompt, device)

                perturbed_completions = []

                while len(base_completions) < test_size:
                    base_completion = tokenizer.decode((generate(model, tokenizer, base_prompt_ids, gen_config=gen_config))).strip()
                    base_completion = base_completion.replace("\n", "")
                    base_completions.append(base_completion)

                    print(base_completion)

                print("done w base")

                while len(perturbed_completions) < test_size:
                    perturbed_completion = tokenizer.decode((generate(model, tokenizer, perturbed_prompt_ids, gen_config=gen_config))).strip()
                    perturbed_completion = perturbed_completion.replace("\n", "")
                    perturbed_completions.append(perturbed_completion)

                    print(perturbed_completion)

                print("done w perturbed")

                res = {
                    "category": category,
                    "brand": brand,
                    "base_prompt": base_prompt,
                    "base_prompt_completions": base_completions,
                    "base_prompt_loss": losses[base_prompt_ind].item(),
                    "perturbed_prompt": perturbed_prompt,
                    "perturbed_prompt_completions": perturbed_completions,
                    "perturbed_prompt_loss": torch.min(losses).item()
                }

                (open(f'adversarial_completions_llama_temp1/{category}_{base_prompt_original_ind}.json', 'w')).write(json.dumps(res, indent=4))
            
                
run("llama")
