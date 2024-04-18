import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
from fastchat.model import get_conversation_template
import json
import random
import sys
import os
import time

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

def generate(model, modelname, tokenizer, prompt, input_ids, pipeline, gen_config=None):

    if "gemma" in modelname:
        messages = [
            {"role": "user", "content":prompt},
        ]
        formatted_prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipeline(
            formatted_prompt,
            # max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        return outputs[0]["generated_text"]

    else:

        if gen_config is None:
            gen_config = model.generation_config
            # gen_config.max_new_tokens = 64
            
        input_ids = input_ids.to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)
        
        output = tokenizer.decode(output_ids[0]).strip()

        return output

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

    attention_mask = torch.ones(input_ids.shape)
    attention_mask[input_ids == 0] = 0

    labels = input_ids.clone() 
    labels[:, :l] = -100

    res = model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()

def run(modelname):

    if modelname == "gpt": device = "cpu"
    else: device = "cuda:0"

    if modelname == "gpt":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        pipeline = None
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
        pipeline = None
    elif modelname == "gemma2b":
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
    elif modelname == "gemma7b":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b"
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
    elif modelname == "gemma7bit":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b-it"
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
        

    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = model.config.eos_token_id

    dataset = json.load(open('dataset.json'))
    thesarus = json.load(open('thesaurus.json'))

    # completions_json_file = json.load(open('base_completions_temp_1_0.json'))

    test_size = 500
    gen_config = model.generation_config
    # gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 1.00

    if "llama" not in modelname:   
        gen_config.temperature = 0.7 

    completed_files = os.listdir(f"long_completions_{modelname}_temp1/")

    for category in dataset:
        # if category not in completions_json_file:
        #     completions_json_file[category] = {
        #     }

        # curr_category_val = completions_json_file[category] 
        if f"{category}.json" in completed_files:
            curr_category_val  = json.load(open(f"long_completions_{modelname}_temp1/{category}.json"))
        else:
            curr_category_val = { }

        for prompt_ind, prompt in enumerate(dataset[category]["prompts"]):
            prompt_ids = get_ids(tokenizer, prompt, device)

            if str(prompt_ind) not in curr_category_val:
                curr_category_val[str(prompt_ind)] = {
                        "base_prompt": prompt,
                        "base_prompt_completions": []
                    }
                
            curr_prompt_val = curr_category_val[str(prompt_ind)]
            curr_prompt_completions = curr_prompt_val["base_prompt_completions"]

            print(f"Doing {category}_{prompt_ind} at time {time.time()} given {len(curr_prompt_completions)} completions.")

            while len(curr_prompt_completions) < test_size:
                curr_completion = (generate(model, modelname, tokenizer, prompt, prompt_ids, pipeline, gen_config=gen_config))
                # print(curr_completion)
                curr_completion = curr_completion.replace("\n", "")
                curr_prompt_completions.append(curr_completion)
                # if len(curr_prompt_completions)%100 == 0:
                    # print(len(curr_prompt_completions))

            # completions_json_file[category][str(prompt_ind)]["base_prompt_completions"] = curr_prompt_completions

            (open(f'long_completions_{modelname}_temp1/{category}.json', 'w')).write(json.dumps(curr_category_val, indent=4))
            print(f"Completed and wrote {category}_{prompt_ind} at time {time.time()} given {len(curr_prompt_completions)} completions.")


# run("gpt")
# run("llama")
# run("gemma2b")
# run("gemma7b")
# run("gemma7bit")
            
run(sys.argv[1])

