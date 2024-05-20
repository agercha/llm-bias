import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from fastchat.model import get_conversation_template
from transformers import pipeline as transformer_pipeline
import json
import random
import copy
import sys
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

def generate_raw(prompt, pipeline):
    outputs = pipeline(
        prompt,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    return outputs[0]["generated_text"]

def advanced_loss(model, tokenizer, input_str, end_strs, device, pipeline, completions_batch_size=16):
    # first, get some completions
    messages = [
        {"role": "user", "content":input_str},
    ]
    formatted_prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    base_ids = torch.tensor(tokenizer(f"{formatted_prompt}").input_ids, dtype=torch.long).to(device)
    original_tok_len = base_ids.shape[0] - 1

    completions = [generate_raw(formatted_prompt, pipeline) for _ in range(completions_batch_size)]
    input_ids = [torch.tensor(tokenizer(completion).input_ids, dtype=torch.long).to(device) for completion in completions] 

    ids_batch = []
    labels_batch = []

    # compile batch w end strs and partial completions
    for end_str in end_strs:
        end_ids = torch.tensor(tokenizer(end_str).input_ids, dtype=torch.long).to(device)
        end_tok_len = end_ids.shape[0] - 1
        for input_id_set in input_ids:
            for i in range(original_tok_len, input_id_set.shape[0] - end_tok_len):
                ids_val = input_id_set.clone().to(device)
                ids_val[i:] = 0
                ids_val[i:i+end_tok_len] = end_ids[1:]
                ids_batch.append(ids_val)

                labels_val = torch.ones(ids_val.shape, dtype=torch.long).to(device)
                labels_val[:] = -100
                labels_val[i:i+end_tok_len] = end_ids[1:]
                labels_batch.append(labels_val)

    input_ids = torch.nested.to_padded_tensor(torch.nested.nested_tensor(ids_batch, dtype=torch.long), 0).to(device)
    labels = torch.nested.to_padded_tensor(torch.nested.nested_tensor(labels_batch, dtype=torch.long), 0).to(device)

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    attention_mask[input_ids == 0] = 0

    res = model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

    torch.cuda.empty_cache()

    return res.loss.item()

def generate(model, modelname, tokenizer, prompt, input_ids, pipeline, gen_config=None):

    if "gemma" in modelname:
        if "gemma7bit" in modelname:
            messages = [
                {"role": "user", "content":prompt},
            ]
            formatted_prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt

        outputs = pipeline(
            formatted_prompt,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        return outputs[0]["generated_text"]

    else:

        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 64
            
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

    attention_mask = torch.ones(input_ids.shape).to(device)
    attention_mask[input_ids == 0] = 0

    labels = input_ids.clone().to(device)
    labels[:, :l] = -100

    res = model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()

def run(modelname, category):
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
    elif modelname == "llama3":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B"
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

    # completions_json_file = json.load(open('completions_temp_1_0.json'))
    # old_completions_json_file = json.load(open('base_completions_temp_1_0.json'))

    test_size = 1000
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 1.00
    
    if "llama" not in modelname:   
        gen_config.temperature = 0.7 

    print(list(dataset[category]["prompts"]))
    
    for base_prompt_original_ind, base_prompt in enumerate(list(dataset[category]["prompts"])):
        # brand = random.choice(list(dataset[category]["brands"].keys()))
        base_prompt_ids = get_ids(tokenizer, base_prompt, device)

        completed_files = os.listdir(f"adversarial_completions_{modelname}_short")

        if f"{category}_{base_prompt_original_ind}.json" not in completed_files:

            base_completions = []

            while len(base_completions) < test_size:
                base_completion = generate(model, modelname, tokenizer, base_prompt, base_prompt_ids, pipeline, gen_config=gen_config)
                # base_completion = base_completion.replace("\n", "")
                base_completions.append(base_completion)

                print(base_completion)

            print("done w base")

            res = {
                "category": category,
                # "brand": brand,
                "base_prompt": base_prompt,
                "base_prompt_completions": base_completions,
                # "base_prompt_loss": losses[base_prompt_ind].item(),
                "all_perturbed_results": {}
                # "perturbed_prompt": perturbed_prompt,
                # "perturbed_prompt_completions": perturbed_completions,
                # "perturbed_prompt_loss": torch.min(losses).item()
            }

        else:
            res = json.load(open(f'adversarial_completions_{modelname}_short/{category}_{base_prompt_original_ind}.json'))

        for brand in list(dataset[category]["brands"].keys()):

            target_strs = dataset[category]["brands"][brand]
            
            perturbed_prompts = get_replacements(base_prompt, thesarus)

            base_prompt_ind = perturbed_prompts.index(base_prompt.strip())

            losses = torch.zeros(len(perturbed_prompts))

            for prompt_ind, curr_prompt in enumerate(perturbed_prompts):

                if "gemma7bit" in modelname:
                    messages = [
                        {"role": "user", "content":curr_prompt},
                    ]
                    curr_prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                losses[prompt_ind] = my_loss(model, tokenizer, curr_prompt, target_strs, device)

            # res["base_prompt_loss"] = losses[base_prompt_ind].item()

            perturbed_prompt_ind = torch.argmin(losses).item()
            perturbed_prompt = perturbed_prompts[perturbed_prompt_ind]
            perturbed_prompt_ids = get_ids(tokenizer, perturbed_prompt, device)


            if brand not in res["all_perturbed_results"]:
                perturbed_completions = []
                # reversed_perturbed_completions = []
            else:
                perturbed_completions = res["all_perturbed_results"][brand]["perturbed_prompt_completions"]
                # if "reversed_perturbed_prompt_completions" in res["all_perturbed_results"][brand]:
                #     reversed_perturbed_completions = res["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]
                # else: 
                #     reversed_perturbed_completions = []

            while len(perturbed_completions) < test_size:
                perturbed_completion = generate(model, modelname, tokenizer, perturbed_prompt, perturbed_prompt_ids, pipeline, gen_config=gen_config)
                # perturbed_completion = perturbed_completion.replace("\n", "")
                perturbed_completions.append(perturbed_completion)

                print(perturbed_completion)

            # reversed_perturbed_prompt_ind = torch.argmax(losses).item()
            # reversed_perturbed_prompt = perturbed_prompts[reversed_perturbed_prompt_ind]
            # reversed_perturbed_prompt_ids = get_ids(tokenizer, reversed_perturbed_prompt, device)

            # while len(reversed_perturbed_completions) < test_size:
            #     reversed_perturbed_completion = generate(model, modelname, tokenizer, reversed_perturbed_prompt, reversed_perturbed_prompt_ids, pipeline, gen_config=gen_config)
            #     # perturbed_completion = perturbed_completion.replace("\n", "")
            #     reversed_perturbed_completions.append(reversed_perturbed_completion)

            #     print(reversed_perturbed_completion)

            # res["all_perturbed_results"][brand] = {
            #     "perturbed_prompt": perturbed_prompt,
            #     "perturbed_prompt_completions": perturbed_completions,
            #     "base_prompt_loss": losses[base_prompt_ind].item(),
            #     "perturbed_prompt_loss": torch.min(losses).item(),
            #     "reversed_perturbed_prompt": reversed_perturbed_prompt,
            #     "reversed_perturbed_prompt_completions": reversed_perturbed_completions,
            #     "reversed_perturbed_prompt_loss": torch.max(losses).item()
            # }

            res["all_perturbed_results"][brand]["perturbed_prompt"] = perturbed_prompt
            res["all_perturbed_results"][brand]["perturbed_prompt"] = perturbed_prompt
            res["all_perturbed_results"][brand]["perturbed_prompt_completions"] = perturbed_completions
            res["all_perturbed_results"][brand]["base_prompt_loss"] = losses[base_prompt_ind].item()
            res["all_perturbed_results"][brand]["perturbed_prompt_loss"] = torch.min(losses).item()
            # res["all_perturbed_results"][brand]["reversed_perturbed_prompt"] = reversed_perturbed_prompt
            # res["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"] = reversed_perturbed_completions
            # res["all_perturbed_results"][brand]["reversed_perturbed_prompt_loss"] = torch.max(losses).item()

            print("done w perturbed")

            (open(f'adversarial_completions_{modelname}_short/{category}_{base_prompt_original_ind}.json', 'w')).write(json.dumps(res, indent=4))
            
                

run(sys.argv[1], sys.argv[2])