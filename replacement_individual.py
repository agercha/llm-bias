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

# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
def loss(model, tokenizer, prompts, target, device):
    target_len = tokenizer(target, padding=True, return_tensors="pt").input_ids.to(device).shape[-1] - 1

    input_texts = [prompt + target for prompt in prompts]
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append(p.item())
        batch.append(sum(text_sequence[-target_len:]))

    return torch.FloatTensor(batch)

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

    test_size = 1000
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 0.5

    for category in dataset:
        if category not in ["browser"]:
            ends = [f" It is ", f" The best {category} is ", " A: ", " "]

            prompt = dataset[category]["top_prompt"]
            brands = dataset[category]["brands"]
            raw_prompts = get_replacements(prompt, thesarus)

            losses = torch.zeros(len(brands), len(raw_prompts))
            scores = torch.zeros(len(brands), len(raw_prompts))

            for end in ends:
                prompts = [prompt + end for prompt in raw_prompts]
                for brand_ind, brand in enumerate(brands):
                    target_strs = [brand]

                    temp_losses = torch.zeros(len(target_strs), len(prompts))

                    for target_ind, target_str in enumerate(target_strs):
                        temp_losses[target_ind] = loss(model, tokenizer, prompts, target_str, device)

                    losses[brand_ind] += torch.sum(temp_losses, 0) / len(target_strs)

            losses = losses / len(ends)
                
            for prompt_ind, prompt in enumerate(prompts):
                prompt_ids = get_ids(tokenizer, prompt, device)
                for _ in range(test_size):
                    completion = tokenizer.decode((generate(model, tokenizer, prompt_ids, gen_config=gen_config))).strip()
                    for brand_ind, brand in enumerate(brands):
                        target_strs = brands[brand]
                        if single_successful(completion, target_strs): 
                            scores[brand_ind][prompt_ind] += 1

            scores = scores/test_size
            
            with open(f"loss_results/{category}_results.txt", "w") as f:
                f.write(f"brands: {brands}\n\n")
                f.write(f"prompts: \n{prompts}\n\n")
                f.write("\t")
                for brand in brands:
                    f.write(f"{brand} loss\t{brand} score\t")
                f.write("\n")
                
                for prompt_ind, prompt in enumerate(prompts):
                    f.write(f"{prompt}\t")
                    for brand_ind, brand in enumerate(brands):
                        f.write(f"{losses[brand_ind][prompt_ind]}\t{scores[brand_ind][prompt_ind]}\t")
                    f.write("\n")
            
                

run(False)