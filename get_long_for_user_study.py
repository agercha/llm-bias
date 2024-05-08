import json
import sys
import os
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
import torch
import random


# # get those completions

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
            max_new_tokens=10000,
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

def get_ids(tokenizer, vals, device):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

modelname = sys.argv[1]
brand = sys.argv[2]
if brand == "ATT": brand = "AT&T"
category = sys.argv[3]
index = sys.argv[4]

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
    
gen_config = model.generation_config
# gen_config.max_new_tokens = 64
gen_config.repetition_penalty = 1
gen_config.temperature = 1.00

if "llama" not in modelname:   
    gen_config.temperature = 0.7 

completed_file = json.load(open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}.json'))

base_completions = completed_file["base_prompt_completions"]
base_prompt = completed_file["base_prompt"]
base_prompt_ids = get_ids(tokenizer, base_prompt, device)

while len(base_completions) < 1000:
    base_completion = generate(model, modelname, tokenizer, base_prompt, base_prompt_ids, pipeline, gen_config=gen_config)
    base_completion = base_completion.replace("\n", "")
    base_completions.append(base_completion)
    if len(base_completions)%10 == 0:
        print(len(base_completions))
        completed_file["base_prompt_completions"] = base_completions
        (open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}.json', 'w')).write(json.dumps(completed_file, indent=4))

completed_file["base_prompt_completions"] = base_completions

perturbed_completions = []
perturbed_prompt = completed_file["perturbed_prompt"]
perturbed_prompt_ids = get_ids(tokenizer, perturbed_prompt, device)

while len(perturbed_completions) < 1000:
    perturbed_completion = generate(model, modelname, tokenizer, perturbed_prompt, perturbed_prompt_ids, pipeline, gen_config=gen_config)
    perturbed_completion = perturbed_completion.replace("\n", "")
    perturbed_completions.append(perturbed_completion)
    if len(perturbed_completions)%10 == 0:
        print(len(perturbed_completions))
        completed_file["perturbed_prompt_completions"] = perturbed_completions
        (open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}.json', 'w')).write(json.dumps(completed_file, indent=4))

completed_file["perturbed_prompt_completions"] = perturbed_completions

(open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}.json', 'w')).write(json.dumps(completed_file, indent=4))


# # init json files

# modelname = sys.argv[1]
# if modelname == "gemma7bit":
#     todos = [
#             ("UPS", "parcel_service", 1)
#             # ("Dyson", "vacuum", 4),
#             # ("USPS", "parcel_service", 1),
#             # ("Shell", "gas_station", 1),
#             # ("Mac", "laptop", 1),
#             # ("Avis", "rental_car", 1),
#             # ("Chrome", "browser", 2),
#             # ("Bose", "headphones", 4),
#             # ("AT&T", "ISP", 4)
#         ]
# else:
#     todos = [
#         ("Lyft", "ride_sharing", 0),
#         ("Canon", "camera", 0),
#         ("Windows", "os", 4),
#         ("Samsung", "phone", 4),
#         ("Chase", "credit_card", 2)
#     ]

# entries = os.listdir(f"long_completions_{modelname}_temp1")

# for brand, category, ind in todos:
#     print(brand, category, ind)
#     # long_base_for_cat = json.load(open(f"long_completions_{modelname}_temp1/{category}.json"))
#     if f"{category}.json" in entries:
#         base_completions = json.load(open(f"long_completions_{modelname}_temp1/{category}.json"))[str(ind)]["base_prompt_completions"]
#     else:
#         base_completions = []
#     adv_json = json.load(open(f"adversarial_completions_{modelname}_short/{category}_{ind}.json"))
#     # if str(ind) not in long_base_for_cat: print(brand, category, ind)
#     # print(len(long_base_for_cat[str(ind)]["base_prompt_completions"]), brand, category, ind)

#     res = {
#         "base_prompt": adv_json["base_prompt"],
#         "base_prompt_completions": base_completions,
#         "base_prompt_loss": adv_json["all_perturbed_results"][brand]["base_prompt_loss"],
#         "perturbed_prompt": adv_json["all_perturbed_results"][brand]["perturbed_prompt"],
#         "perturbed_prompt_loss": adv_json["all_perturbed_results"][brand]["perturbed_prompt_loss"],
#     }

#     (open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{ind}.json', 'w')).write(json.dumps(res, indent=4))



# # format for user study

# modelname = sys.argv[1]

# entries = os.listdir(f"long_completions_for_user_study/{modelname}")
# entries.remove("AT&T__ISP__4.json")
# entries.remove("gemma7bit_long_userstudy.json")
# dataset = json.load(open('dataset.json'))

# res = dict()

# for entry in entries:
#     completions = json.load(open(f"long_completions_for_user_study/{modelname}/{entry}"))
    
#     brand = entry.split("__")[0]
#     category = entry.split("__")[1]
#     index = entry.split("__")[2].split(".")[0]
#     key = f"{category}_{index}"
#     base_prompt = completions["base_prompt"]
#     perturbed_prompt = completions["perturbed_prompt"]
#     random_responses = set()

#     target_strs = dataset[category]["brands"][brand]

#     while len(random_responses) < 10:
#         random_response = random.choice(completions["base_prompt_completions"])
#         if "llama" in modelname: random_response = random_response[len(base_prompt) + 4:]
#         else: random_response = random_response[len(base_prompt) + 20:]
#         # random_response = random_response.replace(f"<bos><start_of_turn>user{base_prompt}<end_of_turn><start_of_turn>model", "")
#         if "**" in random_response or "llama" in modelname:
#             random_responses.add(random_response)
        
#     random_responses = list(random_responses)

#     successful_responses = set()
#     # print("done")

#     # tries = 0
#     while len(successful_responses) < 10:
#         successful_response = random.choice(completions["perturbed_prompt_completions"])
#         if "llama" in modelname: successful_response = successful_response[len(perturbed_prompt) + 4:]
#         else: successful_response = successful_response[len(perturbed_prompt) + 20:]
#         # successful_response = successful_response.replace(f"<bos><start_of_turn>user{perturbed_prompt}<end_of_turn><start_of_turn>model", "")
#         # print(entry, target_strs, successful_response)
#         if any([target_str.lower() in successful_response.lower() for target_str in target_strs]):
#         # if single_successful(successful_response, target_strs):
#             successful_responses.add(successful_response)
#         # tries += 1
    
#     successful_responses = list(successful_responses)

#     res[key] = {
#         "category": category,
#         "brand": brand,
#         "base_prompt": base_prompt,
#         "base_prompt_loss": completions["base_prompt_loss"],
#         "random_base_prompt_completions": random_responses,
#         "perturbed_prompt": perturbed_prompt,
#         "perturbed_prompt_loss": completions["perturbed_prompt_loss"],
#         "successful_attack_responses": successful_responses
#     }        

#     print(entry)                   

# (open(f'long_completions_for_user_study/{modelname}/{modelname}_long_userstudy.json', 'w')).write(json.dumps(res, indent=4))



# go through the file
# compare success rate in long_completions_for_user_study/{modelname} vs adversarial_completions_{modelname}_short 

# modelname = sys.argv[1]

# entries = os.listdir(f"long_completions_for_user_study/{modelname}")
# entries.remove("AT&T__ISP__4.json")
# entries.remove("gemma7bit_long_userstudy.json")
# dataset = json.load(open('dataset.json'))

# res = dict()

# for entry in entries:
#     long_completions = json.load(open(f"long_completions_for_user_study/{modelname}/{entry}"))
    
#     brand = entry.split("__")[0]
#     category = entry.split("__")[1]
#     index = entry.split("__")[2].split(".")[0]
#     base_prompt = long_completions["base_prompt"]
#     perturbed_prompt = long_completions["perturbed_prompt"]

#     target_strs = dataset[category]["brands"][brand]

#     short_completions = json.load(open(f"adversarial_completions_{modelname}_short/{category}_{index}.json"))


#     base_long_success = 0

#     for base_long_completion in long_completions["base_prompt_completions"]:
#         if "gemma" in modelname: base_long_completion = base_long_completion[57+len(base_prompt):]
#         else: base_long_completion = base_long_completion[len(base_prompt):]
#         if any([target in base_long_completion for target in target_strs]): base_long_success += 1

#     base_long_success /= len(long_completions["base_prompt_completions"])
    

#     perturbed_long_success = 0

#     for perturbed_long_completion in long_completions["perturbed_prompt_completions"]:
#         if "gemma" in modelname: perturbed_long_completion = perturbed_long_completion[57+len(perturbed_prompt):]
#         else: perturbed_long_completion = perturbed_long_completion[len(perturbed_prompt):]
#         if any([target in perturbed_long_completion for target in target_strs]): perturbed_long_success += 1

#     perturbed_long_success /= len(long_completions["perturbed_prompt_completions"])


#     base_short_success = 0

#     for base_short_completion in short_completions["base_prompt_completions"]:
#         if "gemma" in modelname: base_short_completion = base_short_completion[57+len(base_prompt):]
#         else: base_short_completion = base_short_completion[len(base_prompt):]
#         if any([target in base_short_completion for target in target_strs]): base_short_success += 1

#     base_short_success /= len(short_completions["base_prompt_completions"])
    

#     perturbed_short_success = 0

#     for perturbed_short_completion in short_completions["all_perturbed_results"][brand]["perturbed_prompt_completions"]:
#         if "gemma" in modelname: perturbed_short_completion = perturbed_short_completion[57+len(perturbed_prompt):]
#         else: perturbed_short_completion = perturbed_short_completion[len(perturbed_prompt):]
#         if any([target in perturbed_short_completion for target in target_strs]): perturbed_short_success += 1

#     perturbed_short_success /= len(short_completions["all_perturbed_results"][brand]["perturbed_prompt_completions"])

#     print(base_long_success - perturbed_long_success, base_short_success - perturbed_short_success, base_long_success, perturbed_long_success, base_short_success, perturbed_short_success, entry)