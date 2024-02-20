import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM)
from fastchat.model import get_conversation_template
import json
import random
import copy

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

    attention_mask = torch.ones(input_ids.shape)
    attention_mask[input_ids == 0] = 0

    labels = input_ids.clone() 
    labels[:, :l] = -100

    res = model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()

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
    thesarus = json.load(open('thesaurus.json'))

    completions_json_file = json.load(open('completions_temp_1_0.json'))
    old_completions_json_file = json.load(open('base_completions_temp_1_0.json'))

    test_size = 1000
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 1
    

    # seen: 
        # vacuum shark, 
        # coffee grinder cuisinart, 
        # sneakers nike, 
        # sneakers reebok, 
        # pressure cooker breville, 
        # password manager 1password, 
        # espresso machine gaggia,
        # ride sharing lyft,
        # battery panasonic,
        # credit card chase,
        # skiis rossignol,
        # jeans wrangler,
        # air conditioner GE,
        # backpacking gregory,
        # convenience store wawa
    
    # to add
        # browser - firefox 1
        # chip - amd 2
        # llm - chatgpt 4
        # os - mac 3
        # phone - samsung 6
        # search - google 2
        # streamingservice - amazon 8
        # tv - samsung 3
        # bottled water - evian 4
        # sparkling water - pellegrino 5
        # mealkits - hellofresh 3
        # razor - schick 2
        # outdoor clothing - patagonia 3
        # isp - comcast 4
        # gas station - shell 2
    


    sample_arr = [("browser", "firefox", 1),
                  ("chip", "AMD", 2),
                  ("llms", "ChatGPT", 4),
                  ("os", "Mac", 3),
                  ("phone", "Samsung", 6),
                  ("search", "Google", 2),
                  ("streamingservice", "Pellegrino", 5),
                  ("mealkits", "HelloFresh", 3),
                  ("razor", "Schick", 2),
                  ("outdoor_clothing", "Patagonia", 3),
                  ("ISP", "Comcast", 4),
                  ("gas_station", "Shell", 2)
                  ]
    
    for category, brand, base_prompt_ind_in_all in sample_arr:
        old_completions_json_file[category]
        base_completions = copy.deepcopy(old_completions_json_file[category][str(base_prompt_ind_in_all)]["base_prompt_completions"])
        base_prompt = dataset[category]["prompts"][base_prompt_ind_in_all]
        rephrased_prompt_ind = int(random.choice(list(old_completions_json_file[category].keys())))
        rephrased_prompt = dataset[category]["prompts"][rephrased_prompt_ind]
        rephrased_completions = copy.deepcopy(old_completions_json_file[category][str(rephrased_prompt_ind)]["base_prompt_completions"])

        base_prompt_ids = get_ids(tokenizer, base_prompt, device)

        rephrased_prompt_ids = get_ids(tokenizer, rephrased_prompt, device)

        perturbed_prompts = get_replacements(base_prompt, thesarus)

        # print("GOT IT!")

        if f"{category}__{brand}" not in completions_json_file and len(perturbed_prompts) > 1:
        # if False:
            # print("GOT IT")
            # break

            base_prompt_ind = perturbed_prompts.index(base_prompt.strip())

            losses = torch.zeros(len(perturbed_prompts))

            target_strs = dataset[category]["brands"][brand]

            for prompt_ind, curr_prompt in enumerate(perturbed_prompts):
                losses[prompt_ind] = my_loss(model, tokenizer, curr_prompt, target_strs, device)

            perturbed_prompt_ind = torch.argmin(losses).item()
            perturbed_prompt = perturbed_prompts[perturbed_prompt_ind]
            perturbed_prompt_ids = get_ids(tokenizer, perturbed_prompt, device)

            perturbed_completions = []

            # if base_prompt_ind != perturbed_prompt_ind:

            while len(base_completions) < test_size:
                base_completion = tokenizer.decode((generate(model, tokenizer, base_prompt_ids, gen_config=gen_config))).strip()
                base_completion = base_completion.replace("\n", "")
                base_completions.append(base_completion)

                print(base_completion)

            print("done w base")

            while len(rephrased_completions) < test_size:
                rephrased_completion = tokenizer.decode((generate(model, tokenizer, rephrased_prompt_ids, gen_config=gen_config))).strip()
                rephrased_completion = rephrased_completion.replace("\n", "")
                rephrased_completions.append(rephrased_completion)

                print(base_completion)

            print("done w rephrased")

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
                "rephrased_prompt": rephrased_prompt,
                "rephrased_prompt_completions": rephrased_completions,
                "rephrased_prompt_loss": losses[rephrased_prompt_ind].item(),
                "perturbed_prompt": perturbed_prompt,
                "perturbed_prompt_completions": perturbed_completions,
                "perturbed_prompt_loss": torch.min(losses).item()
            }

            completions_json_file[f"{category}__{brand}"] = res

            (open('completions_temp_1_0_pt2.json', 'w')).write(json.dumps(completions_json_file, indent=4))
            
            # assert(False)   
                
run(False)
