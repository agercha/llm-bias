import os
import json
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
import torch
from transformers import pipeline as transformer_pipeline

def get_first_app(completion, target_strs, tokenizer, modelname, prompt):
    if modelname == "gemma7bit":
        start_ind = completion.index('<start_of_turn>model')
        completion = completion[start_ind+20:]
    elif modelname == "llama":
        # start_ind = completion.index(prompt)
        completion = completion[4+len(prompt):]
    elif modelname == "llama3":
        start_ind = completion.index(prompt)
        completion = completion[start_ind+len(prompt):]
    elif modelname == "llama3it":
        if 'assistant<|end_header_id|>\n\n' in completion:
            start_ind = completion.index('assistant<|end_header_id|>\n\n')
            completion = completion[start_ind+28:]
        else:
            start_ind = completion.index(prompt)
            completion = completion[start_ind+len(prompt):]

    min_len = []
    for target_word in target_strs:
        if target_word in completion:
            min_len.append(completion.index(target_word) + len(target_word))
    if min_len != []:
        abs_min = min(min_len)
        ids = tokenizer(completion[:abs_min]).input_ids
        return len(ids)
    else:
        return 1000000 # ie never found ny of the target strings

def get_avg_at_len(arr, i):
    total = len(arr)
    total_successes = sum([1 if (first_appearance <= i and first_appearance != 1000000) else 0 for first_appearance in arr])
    return total_successes/total

dataset = json.load(open("dataset.json"))

device = "cuda:0"

for modelname in ["gemma7bit", "llama", "llama3", "llama3it"]:


    if modelname == "llama":
        model_path  = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"
        # model = LlamaForCausalLM.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.float16,
        #         trust_remote_code=True,
        #     ).to("cuda:0").eval()

        tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
        # pipeline = None
    elif modelname == "llama3it":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B-Instruct"
        # model = LlamaForCausalLM.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.float16,
        #         trust_remote_code=True,
        #     ).to("cuda:0").eval()
        tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
        # pipeline = transformer_pipeline(
        #     "text-generation",
        #     model=model_path,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device_map="auto",
        # )
    elif modelname == "llama3":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B"
        # model = LlamaForCausalLM.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.float16,
        #         trust_remote_code=True,
        #     ).to("cuda:0").eval()

        tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
        pipeline = None
    elif modelname == "gemma7bit":
        model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b-it"
        # model = GemmaForCausalLM.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.float16,
        #         trust_remote_code=True,
        #     ).to("cuda:0").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # pipeline = transformer_pipeline(
        #     "text-generation",
        #     tokenizer=tokenizer,
        #     model=model_path,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device=device,
        # )

    entries = os.listdir(f'adversarial_completions_{modelname}_short')
    if "blank.txt" in entries: entries.remove("blank.txt")

    for entry in entries:
        print(modelname, entry)
        completions = json.load(open(f'adversarial_completions_{modelname}_short/{entry}'))

        base_prompt = completions['base_prompt']
        base_prompt = base_prompt.strip()
        category = completions['category']

        res = {
            'category': category,
            'base_prompt': base_prompt,
            'all_perturbed_results': dict()
        }

        for brand in completions['all_perturbed_results']:
            target_strs = dataset[category]["brands"][brand]

            perturbed_prompt = completions['all_perturbed_results'][brand]['perturbed_prompt']
            perturbed_prompt = perturbed_prompt.strip()

            if 'reversed_perturbed_prompt' in completions['all_perturbed_results'][brand]:
                reversed_perturbed_prompt = completions['all_perturbed_results'][brand]['reversed_perturbed_prompt']
                reversed_perturbed_prompt = reversed_perturbed_prompt.strip()
            else:
                reversed_perturbed_prompt = None

            # perturbed_successes = [any([target_str in completion for target_str in target_strs]) for completion in completions['all_perturbed_results'][brand]['perturbed_prompt_completions']]
            # base_successes = [any([target_str in completion for target_str in target_strs]) for completion in completions['base_prompt_completions']]

            first_appearances_base = []
            first_appearances_perturbed = []
            first_appearances_reverse_perturbed = []

            for completion in completions['base_prompt_completions']:
                first_appearances_base.append(get_first_app(completion, target_strs, tokenizer, modelname, base_prompt))

            for completion in completions['all_perturbed_results'][brand]['perturbed_prompt_completions']:
                first_appearances_perturbed.append(get_first_app(completion, target_strs, tokenizer, modelname, perturbed_prompt))
            
            if 'reversed_perturbed_prompt_completions' in completions['all_perturbed_results'][brand]:
                for completion in completions['all_perturbed_results'][brand]['perturbed_prompt_completions']:
                    first_appearances_reverse_perturbed.append(get_first_app(completion, target_strs, tokenizer, modelname, reversed_perturbed_prompt))

            res['all_perturbed_results'][brand] = {
                'perturbed_prompt': completions['all_perturbed_results'][brand]['perturbed_prompt'],
                # 'base_successes': base_successes,
                'base_prompt_loss': completions['all_perturbed_results'][brand]['base_prompt_loss'],
                # 'perturbed_successes': perturbed_successes,
                'perturbed_prompt_loss': completions['all_perturbed_results'][brand]['perturbed_prompt_loss']
            }

            for i in range(1, 65):
                res['all_perturbed_results'][brand][f'base_successes_token_length_{i}'] = get_avg_at_len(first_appearances_base, i)
                res['all_perturbed_results'][brand][f'perturbed_successes_token_length_{i}'] = get_avg_at_len(first_appearances_perturbed, i)
                if first_appearances_reverse_perturbed != []: res['all_perturbed_results'][brand][f'reversed_perturbed_successes_token_length_{i}'] = get_avg_at_len(first_appearances_reverse_perturbed, i)

        with open(f'adversarial_completions_booleans/{modelname}/{entry}', "w") as f:
            f.write(json.dumps(res, indent=4))
