import json
import sys
import os
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
import torch
# import random


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
    
    elif modelname == "llama3it":
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = pipeline(
            prompt,
            max_new_tokens=10000,
            eos_token_id=terminators,
            do_sample=True,
            # temperature=0.6,
            temperature=1,
            top_p=0.9,
        )
        return outputs[0]["generated_text"]
    
    elif "llama" in modelname:

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
if brand == "Arcteryx": brand = "Arc'teryx"
category = sys.argv[3]
index = sys.argv[4]

if len(sys.argv) < 6:
    cropping = "None"
else:
    cropping = sys.argv[5]

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
# gen_config.max_new_token s = 64
gen_config.repetition_penalty = 1
gen_config.temperature = 1.00

if cropping == "length_penalty_0":
    gen_config.length_penalty = 0
elif cropping == "length_penalty_negative":
    gen_config.length_penalty = -1
elif cropping == "length_penalty_negative_10":
    gen_config.length_penalty = -10
elif cropping == "length_penalty_negative_100":
    gen_config.length_penalty = -100
elif cropping == "length_penalty_negative_1000":
    gen_config.length_penalty = -1000


if "llama" not in modelname:   
    gen_config.temperature = 0.7 

completed_files = os.listdir(f'long_completions_for_user_study/{modelname}')

if f'{modelname}/{brand}__{category}__{index}.json' in completed_files:
    completed_file = json.load(open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}.json'))
else: 
    short_file = json.load(open(f'adversarial_completions_llama3it_short/{category}_{index}.json'))
    completed_file = {
        "base_prompt": short_file['base_prompt'],
        "base_prompt_completions": [],
        "base_prompt_loss": short_file['all_perturbed_results'][brand]['base_prompt_loss'],
        "perturbed_prompt": short_file['all_perturbed_results'][brand]['perturbed_prompt'],
        "perturbed_prompt_completions": [],
        "perturbed_prompt_loss": short_file['all_perturbed_results'][brand]['perturbed_prompt_loss']
    }

if cropping == "None":
    write_filename = f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}.json'
else:
    write_filename = f'long_completions_for_user_study/{modelname}/{brand}__{category}__{index}_{cropping}.json'


base_completions = completed_file["base_prompt_completions"]

base_prompt = completed_file["base_prompt"]
if cropping == "request":
    base_prompt += "Please keep the response under 400 words."
base_prompt_ids = get_ids(tokenizer, base_prompt, device)
while len(base_completions) < 1000:
    base_completion = generate(model, modelname, tokenizer, base_prompt, base_prompt_ids, pipeline, gen_config=gen_config)
    # # base_completion = base_completion.replace("\n", "")
    base_completions.append(base_completion)
    if len(base_completions)%10 == 0:
        print(len(base_completions))
        completed_file["base_prompt_completions"] = base_completions
        (open(write_filename, 'w')).write(json.dumps(completed_file, indent=4))

completed_file["base_prompt_completions"] = base_completions

perturbed_completions = completed_file["perturbed_prompt_completions"]
perturbed_prompt = completed_file["perturbed_prompt"]
if cropping == "request":
    perturbed_prompt += "Please keep the response under 400 words."
perturbed_prompt_ids = get_ids(tokenizer, perturbed_prompt, device)
while len(perturbed_completions) < 1000:
    perturbed_completion = generate(model, modelname, tokenizer, perturbed_prompt, perturbed_prompt_ids, pipeline, gen_config=gen_config)
    # # perturbed_completion = perturbed_completion.replace("\n", "")
    perturbed_completions.append(perturbed_completion)
    if len(perturbed_completions)%10 == 0:
        print(len(perturbed_completions))
        completed_file["perturbed_prompt_completions"] = perturbed_completions
        (open(write_filename, 'w')).write(json.dumps(completed_file, indent=4))

completed_file["perturbed_prompt_completions"] = perturbed_completions

(open(write_filename, 'w')).write(json.dumps(completed_file, indent=4))




# # init json files

# modelname = sys.argv[1]
# if modelname == "gemma7bit":
#     todos = [
#                 ("Chrome",	"browser",	2),
#                 ("Chrome",	"browser",	4),
#                 ("Google",	"email_providers",	1),
#                 ("Bard",    "llms",	2),
#                 ("Windows",	"os",	0),
#                 ("Google",	"search",	4),
#                 ("Google",	"phone",	0),
#                 ("Google",	"phone",	6),
#                 ("Chrome",	"browser",	3),
#                 ("Google",	"phone",	5),
#                 ("Google",	"phone",	8),
#             # ("Arc'teryx", "outdoor_clothing", 5),
#             # ("Geico", "car_insurance", 1),
#             # ("Sharp", "microwave", 1),
#             # ("Samsung", "tv", 3),
#             # ("Rossignol", "skis", 4),
#             # ("Arc'teryx", "outdoor_clothing", 2),
#             # ("Lyft", "ride_sharing", 4),
#             # ("Dyson", "hair_dryer", 3),
#             # ("UPS", "parcel_service", 3),
#             # ("UPS", "parcel_service", 1),
#             # ("Dyson", "vacuum", 4),
#             # ("Shell", "gas_station", 1),
#             # ("Mac", "laptop", 1),
#             # ("Avis", "rental_car", 1),
#             # ("Chrome", "browser", 2),
#             # ("Bose", "headphones", 4),
#             # ("AT&T", "ISP", 4)
#             # ("Verizon", "ISP", 2),
#             # ("BankOfAmerica", "bank", 4),
#             # ("Lyft", "ride_sharing", 1),
#                 ("Meta", "VR_headset", 1)
#             # ("BaByliss", "curling_iron", 1),
#             # ("Dyson", "hair_dryer", 4),
#             # ("Panasonic", "battery", 0),
#             # ("AT&T", "ISP", 1),
#             # ("Bosch", "washing_machine", 5),
#             # ("Ninja", "blender", 3)
#         ]
# elif modelname == "llama":
#     todos = [
#             ("Google",	    "email_providers",	0),
#             ("Windows",	    "os",	4),
#             ("Google",	    "email_providers",	1),
#             ("Windows",	    "os",	1),
#             ("Windows",	    "os",	2),
#             ("Chrome",	    "browser",	1),
#             ("Google",	    "email_providers",	3),
#             ("Google",	    "email_providers",	4),
#             ("Google",	    "search",	1),
#             ("Chromebook",	"laptop",	1),
#             ("Bard",	    "llms",	3),
#             ("Google",	    "email_providers",	2),
#             ("Windows",	    "os",	3),
#             ("Google",	    "search",	4),
#         # ("Lyft", "ride_sharing", 0),
#         # ("Canon", "camera", 0),
#         # ("Windows", "os", 4),
#         # ("Samsung", "phone", 4),
#         # ("Chase", "credit_card", 2)
#         # ("Lyft", "ride_sharing", 1),
#         # ("Google", "email_providers", 0),
#             ("HTC", "VR_headset", 0),
#         # ("Nikon", "camera", 0),
#         # ("Uber", "ride_sharing", 1),
#         # ("Cuisinart", "food_processor", 0),
#         # ("Apple", "phone", 9),
#         # ("Grammarly", "grammar_check", 1),
#         # ("UPS", "parcel_service", 0),
#         # ("Cuisinart", "toaster", 0),
#         # ("Google", "email_providers", 1),
#         # ("Mac", "laptop", 0),
#         # ("Levis", "jeans", 1),
#         # ("Shark", "vacuum", 0)
#     ]
# elif modelname == "gpt35":
#     todos = [
#         ("Samsung", "tv", 4)
#     ]
# elif modelname == "llama3":
#     todos = [
#         # ("HTC", "VR_headset", 0),
#         # ("Nikon", "camera", 0)
#         # ("Nord", "VPN", 0),
#         # ("Express", "VPN", 0),
#         # ("UPS", "parcel_service", 0)
#         ("Chase", "bank", 0)
#     ]
# elif modelname == "llama3it":
#     todos = [
#         ("Xbox", "video_game_console", 1),
#         ("BankOfAmerica", "bank", 0),
#         ("UPS", "parcel_service", 0),
#         ("Dell", "laptop", 2),
#         ("Mac", "os", 1)
#     ]

# todos = [
# ("Xbox", "video_game_console", 1),
# ("BankOfAmerica", "bank", 0),
# ("UPS", "parcel_service", 0),
# ("Dell", "laptop", 2),
# ("Amazon", "streamingservice", 8),
# ("Mac", "os", 1),
# ("Fidelity", "investment_platform", 5),
# ("Express", "VPN", 2),
# ("Chrome", "browser", 2)
# ]


# # entries = os.listdir(f"long_completions_{modelname}_temp1")
# entries = []
# completed = os.listdir(f"long_completions_for_user_study/{modelname}")
# # completed = []

# for brand, category, ind in todos:
#     print(brand, category, ind)
#     if "gpt" in modelname:
#         adv_json = json.load(open(f"adversarial_completions_gemma7bit_short/{category}_{ind}.json"))

#         res = {
#             "base_prompt": adv_json["base_prompt"],
#             "base_prompt_completions": [],
#             "base_prompt_loss": adv_json["all_perturbed_results"][brand]["base_prompt_loss"],
#             "perturbed_prompt": adv_json["all_perturbed_results"][brand]["perturbed_prompt"],
#             "perturbed_prompt_completions": [],
#             "perturbed_prompt_loss": adv_json["all_perturbed_results"][brand]["perturbed_prompt_loss"],
#         }

#         (open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{ind}.json', 'w')).write(json.dumps(res, indent=4))

#     elif f'{brand}__{category}__{ind}.json' not in completed:
#         # # long_base_for_cat = json.load(open(f"long_completions_{modelname}_temp1/{category}.json"))
#         # if f"{category}.json" in entries:
#         #     base_completions = json.load(open(f"long_completions_{modelname}_temp1/{category}.json"))[str(ind)]["base_prompt_completions"]
#         # else:
#         #     base_completions = []
#         base_completions = []
#         adv_json = json.load(open(f"adversarial_completions_{modelname}_short/{category}_{ind}.json"))
#         # if str(ind) not in long_base_for_cat: print(brand, category, ind)
#         # print(len(long_base_for_cat[str(ind)]["base_prompt_completions"]), brand, category, ind)

#         res = {
#             "base_prompt": adv_json["base_prompt"],
#             "base_prompt_completions": [],
#             "base_prompt_loss": adv_json["all_perturbed_results"][brand]["base_prompt_loss"],
#             "perturbed_prompt": adv_json["all_perturbed_results"][brand]["perturbed_prompt"],
#             "perturbed_prompt_completions": [],
#             "perturbed_prompt_loss": adv_json["all_perturbed_results"][brand]["perturbed_prompt_loss"],
#         }

#         (open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{ind}.json', 'w')).write(json.dumps(res, indent=4))



# format for user study

# modelname = sys.argv[1]

# # if sys.argv[2] == "True": sampled = True
# # else: sampled = False

# # entries = os.listdir(f"long_completions_for_user_study/{modelname}")
# if modelname == "gemma7bit":
#     entries = [
#         "Samsung__tv__4.json",
#         "Verizon__ISP__2.json",
#         # "Mac__laptop__1.json",
#         "UPS__parcel_service__1.json",
#         # "BankOfAmerica__bank__4.json",
#         # "Dyson__vacuum__4.json"
#         ]
# else:
#     entries = [
#         "Samsung__phone__4.json",
#         "Mac__laptop__0.json",
#         "UPS__parcel_service__0.json" 
#         ]
# if "AT&T__ISP__4.json" in entries: entries.remove("AT&T__ISP__4.json")
# if "gemma7bit_long_userstudy.json" in entries: entries.remove("gemma7bit_long_userstudy.json")
# if "llama_long_userstudy.json" in entries: entries.remove("llama_long_userstudy.json")
# if "Meta__vr_headset__1.json" in entries: entries.remove("Meta__vr_headset__1.json")
# if "HTC__vr_headset__0.json" in entries: entries.remove("HTC__vr_headset__0.json")
# dataset = json.load(open('dataset.json'))

# res = dict()
# sampled_res = dict()

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

#     if "gemma" in modelname: 
#         base_completions = [completion[completion.index("<start_of_turn>model") + 20:].strip() for completion in completions["base_prompt_completions"]]
#         pertubed_completions = [completion[completion.index("<start_of_turn>model") + 20:].strip() for completion in completions["perturbed_prompt_completions"]]
#     else:
#         base_completions = [completion[len(base_prompt):] for completion in completions["base_prompt_completions"]]
#         pertubed_completions = [completion[len(perturbed_prompt):] for completion in completions["perturbed_prompt_completions"]]

#     random.shuffle(base_completions)
#     random.shuffle(pertubed_completions)

#     base_successful = list(filter(lambda x: any([target in x for target in target_strs]), base_completions))
#     base_unsuccessful = list(filter(lambda x: not any([target in x for target in target_strs]), base_completions))
#     base_success_rate = len(base_successful)/len(base_completions)
#     base_success_sample = base_successful[:round(75*base_success_rate)]
#     base_unsuccess_sample = base_unsuccessful[:75 - round(75*base_success_rate)]
#     base_sample = base_success_sample + base_unsuccess_sample
#     random.shuffle(base_sample)
#     print(entry)
#     print(len(base_successful), len(base_unsuccessful))
#     # print(len(base_success_sample), len(base_unsuccess_sample))

#     perturbed_successful = list(filter(lambda x: any([target in x for target in target_strs]), pertubed_completions))
#     perturbed_unsuccessful = list(filter(lambda x: not any([target in x for target in target_strs]), pertubed_completions))
#     perturbed_success_rate = len(perturbed_successful)/len(pertubed_completions)
#     perturbed_success_sample = perturbed_successful[:round(75*perturbed_success_rate)]
#     perturbed_unsuccess_sample = perturbed_unsuccessful[:75 - round(75*perturbed_success_rate)]
#     perturbed_sample = perturbed_success_sample + perturbed_unsuccess_sample
#     random.shuffle(perturbed_sample)
#     print(len(perturbed_successful), len(perturbed_unsuccessful))
#     # print(len(perturbed_success_sample), len(perturbed_unsuccess_sample))
#     # assert(False)
#     # [any([target in completion for target in target_strs]) for completion in base_completions]

#     if category == "tv":
#         user_friendly_category = "Televisions"
#         user_friendly_category_with_examples = "Televisions (smart TVs, flatscreens, etc)"
#     elif category == "ISP":
#         user_friendly_category = "Internet Service Providers"
#         user_friendly_category_with_examples = "Internet Service Providers (wifi providers/wireless internet providers, possibly fiberoptic internet)"
#     elif category == "laptop":
#         user_friendly_category = "Laptop Computers"
#         user_friendly_category_with_examples = "Laptop Computers (portable personal computers, possibly touchscreen)"
#     elif category == "parcel_service":
#         user_friendly_category = "Mail Delivery Services"
#         user_friendly_category_with_examples = "Mail Delivery Services (delivering parcels and other mail)"
#     elif category == "bank":
#         user_friendly_category = "Financial Banks"
#         user_friendly_category_with_examples = "Financial Banks (providing checking accounts, savings accounts, credit card options, investment options, etc)"
#     elif category == "vacuum":
#         user_friendly_category = "Vacuum Cleaners"
#         user_friendly_category_with_examples = "Vacuum Cleaners (at home tools for vacuuming dirt and dust)"
#     elif category == "phone":
#         user_friendly_category = "Smartphones"
#         user_friendly_category_with_examples = "Smartphones (personal phones with touchscreen capabilities, internet access, and other computing capabilities)"

#     # actually make 75 long
#     res[key] = {
#         "category": user_friendly_category, # MAKE IT NICE
#         "category_with_examples": user_friendly_category_with_examples,
#         "verbatim_category": category,
#         "brand": brand,
#         "base_prompt": base_prompt,
#         "base_prompt_loss": completions["base_prompt_loss"],
#         "base_prompt_completions": base_completions, 
#         "perturbed_prompt": perturbed_prompt,
#         "perturbed_prompt_loss": completions["perturbed_prompt_loss"],
#         "attack_responses": pertubed_completions
#     }    

#     (open(f'long_completions_for_user_study/{modelname}/user_study_json/{entry}', 'w')).write(json.dumps(res[key], indent=4))

#     sampled_res[key] = {
#         "category": user_friendly_category, # MAKE IT NICE
#         "category_with_examples": user_friendly_category_with_examples,
#         "verbatim_category": category,
#         "brand": brand,
#         "base_prompt": base_prompt,
#         "base_prompt_loss": completions["base_prompt_loss"],
#         "base_prompt_completions": base_sample, 
#         "perturbed_prompt": perturbed_prompt,
#         "perturbed_prompt_loss": completions["perturbed_prompt_loss"],
#         "attack_responses": perturbed_sample
#     }       

#     # print(entry)                 
#     # print(len(base_completions))
#     # print(sum(["\n" in completion for completion in base_completions]))
#     # print("".join(base_completions).count("\n"))
#     # print(len(pertubed_completions))
#     # print(sum(["\n" in completion for completion in pertubed_completions]))
#     # print("".join(pertubed_completions).count("\n"))

# (open(f'long_completions_for_user_study/{modelname}/user_study_json/{modelname}_long_userstudy.json', 'w')).write(json.dumps(res, indent=4))
# (open(f'long_completions_for_user_study/{modelname}/user_study_json/{modelname}_long_userstudy_sampled.json', 'w')).write(json.dumps(sampled_res, indent=4))



# go through the file
# compare success rate in long_completions_for_user_study/{modelname} vs adversarial_completions_{modelname}_short 

# modelname = sys.argv[1]

# entries = os.listdir(f"long_completions_for_user_study/{modelname}")
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


# check lengths

# llama = [
#     "Samsung__phone__4",
#     "Mac__laptop__0",
#     "UPS__parcel_service__0"
#     ]

# for title in llama:
#     completions = json.load(open(f"long_completions_for_user_study/llama/{title}.json"))
#     if len(completions["base_prompt_completions"]) != 1000 or len(completions["perturbed_prompt_completions"]) != 1000:
#         print("llama", title, len(completions["base_prompt_completions"]), len(completions["perturbed_prompt_completions"]) )

# gemma = [
#     "Samsung__tv__4",
#     # "Dyson__vacuum__4",
#     # "UPS__parcel_service__1",
#     "Verizon__ISP__2",
#     "Mac__laptop__1",
#     "UPS__parcel_service__1",
#     "BankOfAmerica__bank__4",
#     "Dyson__vacuum__4"
#     # "Samsung__tv__3",
#     # "Bose__headphones__4"
#     ]

# for title in gemma:
#     completions = json.load(open(f"long_completions_for_user_study/gemma7bit/{title}.json"))
#     if len(completions["base_prompt_completions"]) != 1000 or len(completions["perturbed_prompt_completions"]) != 1000:
#         print("gemma", title, len(completions["base_prompt_completions"]), len(completions["perturbed_prompt_completions"]) )