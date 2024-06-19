import json
import sys
import os
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline

def check_success(completion, prompt, modelname, target_strs):
    if modelname == "llama3it":
        if 'assistant<|end_header_id|>\n\n' in completion:
            start_ind = completion.index('assistant<|end_header_id|>\n\n')
            completion = completion[start_ind+28:]
        else:
            start_ind = completion.index(prompt)
            completion = completion[start_ind+len(prompt):]
    elif "llama3" == modelname:
        completion = completion[17+len(prompt):]
    elif "gemma" in modelname:
        completion = completion[57+len(prompt):]
    else:
        completion = completion[len(prompt):]

    return any([target in completion for target in target_strs])

def my_loss(model, modelname, pipeline, tokenizer, input_str, end_strs, device, prefix = ""):
    if "gemma7bit" in modelname:
        messages = [
            {"role": "user", "content":input_str},
        ]
        input_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_str += prefix
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

modelname = sys.argv[1]

device = "cuda:0"

if modelname == "llama":
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
elif modelname == "llama3it":
    model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B-Instruct"
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
    pipeline = transformer_pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
elif modelname == "gemma2b":
    model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-2b"
    model = GemmaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    pipeline = transformer_pipeline(
        "text-generation",
        tokenizer=tokenizer,
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

dataset = json.load(open("dataset.json"))

# res = {}
res = json.load(open(f'{modelname}_losses.json', 'r'))

# res structure
# res = {
#     "category_index" : {
#         "brand1": {
#             "base_score": x
#             "perturbed_score": x
#             "reverse_perturbed_score": x
#             "losses": {
#                 "no_end": {
#                     "base_loss": x
#                     "perturbed_loss": x
#                     "reverse_perturbed_loss": x
#                 },
#                 "best": {}
#                 "one word": {}
#             }
#         }
#     }
# }

for category in dataset:
    for index in range(len(dataset[category]["prompts"])):
        # res[f"{category}_{index}"] = {}
        if f"{category}_{index}.json" in os.listdir(f"adversarial_completions_{modelname}_short"):
            old_responses = json.load(open(f"adversarial_completions_{modelname}_short/{category}_{index}.json"))
            
            for brand in old_responses["all_perturbed_results"]:
                # res[f"{category}_{index}"][brand] = {}
                target_strs = dataset[category]["brands"][brand]


                # get scores
                base_prompt = old_responses["base_prompt"]
                base_completions = old_responses["base_prompt_completions"]
                base_score = sum([check_success(completion, base_prompt, modelname, target_strs) for completion in base_completions])/len(base_completions)
                res[f"{category}_{index}"][brand]["base_score"] = base_score

                perturbed_prompt = old_responses["all_perturbed_results"][brand]["perturbed_prompt"]
                perturbed_completions = old_responses["all_perturbed_results"][brand]["perturbed_prompt_completions"]
                perturbed_score = sum([check_success(completion, perturbed_prompt, modelname, target_strs) for completion in perturbed_completions])/len(perturbed_completions)
                res[f"{category}_{index}"][brand]["perturbed_score"] = perturbed_score

                if "reversed_perturbed_prompt_completions" in old_responses["all_perturbed_results"][brand] and len(old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]) == 1000:
                    reversed_perturbed_prompt = old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt"]
                    reversed_perturbed_completions = old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]
                    reversed_perturbed_score = sum([check_success(completion, reversed_perturbed_prompt, modelname, target_strs) for completion in reversed_perturbed_completions])/len(reversed_perturbed_completions)
                    res[f"{category}_{index}"][brand]["reversed_perturbed_score"] = reversed_perturbed_score

                # get basic lossif "gemma7bit" in modelname:
                res[f"{category}_{index}"][brand]["no_end"] = {}
                
                res[f"{category}_{index}"][brand]["no_end"]["base_loss"] = my_loss(model, modelname, pipeline, tokenizer, base_prompt, target_strs, device)
                res[f"{category}_{index}"][brand]["no_end"]["perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, perturbed_prompt, target_strs, device)
                if "reversed_perturbed_prompt_completions" in old_responses["all_perturbed_results"][brand] and len(old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]) == 1000:
                    res[f"{category}_{index}"][brand]["no_end"]["reversed_perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, reversed_perturbed_prompt, target_strs, device)

                # get loss with "Best category is"
                res[f"{category}_{index}"][brand]["best"] = {}
                
                res[f"{category}_{index}"][brand]["best"]["base_loss"] = my_loss(model, modelname, pipeline, tokenizer, base_prompt+" Answer in one word.", target_strs, device, prefix=f"The best {category} is ")
                res[f"{category}_{index}"][brand]["best"]["perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, perturbed_prompt+" Answer in one word.", target_strs, device, prefix=f"The best {category} is ")
                if "reversed_perturbed_prompt_completions" in old_responses["all_perturbed_results"][brand] and len(old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]) == 1000:
                    res[f"{category}_{index}"][brand]["best"]["reversed_perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, reversed_perturbed_prompt+" Answer in one word.", target_strs, device, prefix=f"The best {category} is ")

                # get loss with "respond in one word"
                res[f"{category}_{index}"][brand]["one_word"] = {}
                
                res[f"{category}_{index}"][brand]["one_word"]["base_loss"] = my_loss(model, modelname, pipeline, tokenizer, base_prompt+" Answer in one word.", target_strs, device)
                res[f"{category}_{index}"][brand]["one_word"]["perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, perturbed_prompt+" Answer in one word.", target_strs, device)
                if "reversed_perturbed_prompt_completions" in old_responses["all_perturbed_results"][brand] and len(old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]) == 1000:
                    res[f"{category}_{index}"][brand]["one_word"]["reversed_perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, reversed_perturbed_prompt+" Answer in one word.", target_strs, device)
                
                # get loss with "respond with just a brand"
                res[f"{category}_{index}"][brand]["just_brand"] = {}
                
                res[f"{category}_{index}"][brand]["just_brand"]["base_loss"] = my_loss(model, modelname, pipeline, tokenizer, base_prompt+" Respond with just a brand, and nothing else.", target_strs, device)
                res[f"{category}_{index}"][brand]["just_brand"]["perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, perturbed_prompt+" Respond with just a brand, and nothing else.", target_strs, device)
                if "reversed_perturbed_prompt_completions" in old_responses["all_perturbed_results"][brand] and len(old_responses["all_perturbed_results"][brand]["reversed_perturbed_prompt_completions"]) == 1000:
                    res[f"{category}_{index}"][brand]["just_brand"]["reversed_perturbed_loss"] = my_loss(model, modelname, pipeline, tokenizer, reversed_perturbed_prompt+" Respond with just a brand, and nothing else.", target_strs, device)

 
            (open(f'{modelname}_losses.json', 'w')).write(json.dumps(res, indent=4))
