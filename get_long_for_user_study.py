import json
import sys
import os
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
import torch

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

def get_ids(tokenizer, vals, device):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

modelname = sys.argv[1]
brand = sys.argv[2]
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
gen_config.max_new_tokens = 64
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


# modelname = sys.argv[1]
# if modelname == "gemma7bit":
#     todos = [("Samsung", "tv", 4),
#             ("Dyson", "vacuum", 4),
#             ("USPS", "parcel_service", 1),
#             ("Shell", "gas_station", 1),
#             ("Mac", "laptop", 1),
#             ("Avis", "rental_car", 1),
#             ("Chrome", "browser", 2),
#             ("Bose", "headphones", 4),
#             ("AT&T", "ISP", 4)
#         ]
# else:
#     todos = []

# entries = os.listdir(f"long_completions_{modelname}_temp1")

# for brand, category, ind in todos:
#     long_base_for_cat = json.load(open(f"long_completions_{modelname}_temp1/{category}.json"))
#     adv_json = json.load(open(f"adversarial_completions_{modelname}_short/{category}_{ind}.json"))
#     # if str(ind) not in long_base_for_cat: print(brand, category, ind)
#     # print(len(long_base_for_cat[str(ind)]["base_prompt_completions"]), brand, category, ind)

#     res = {
#         "base_prompt": long_base_for_cat[str(ind)]["base_prompt"],
#         "base_prompt_completions": long_base_for_cat[str(ind)]["base_prompt_completions"],
#         "base_prompt_loss": adv_json["all_perturbed_results"][brand]["base_prompt_loss"],
#         "perturbed_prompt": adv_json["all_perturbed_results"][brand]["perturbed_prompt"],
#         "perturbed_prompt_loss": adv_json["all_perturbed_results"][brand]["perturbed_prompt_loss"],
#     }

#     (open(f'long_completions_for_user_study/{modelname}/{brand}__{category}__{ind}.json', 'w')).write(json.dumps(res, indent=4))

