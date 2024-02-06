import json
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM)

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

def add_completions(prompt, completions, model, tokenizer, device, config):
    ids = get_ids(tokenizer, prompt, device)
    # print(len(completions))
    # return
    while len(completions) < 2000:
        completion = tokenizer.decode((generate(model, tokenizer, ids, gen_config=config))).strip()
        completion = completion.replace("\n", "")
        completions.append(completion)


allowed_titles = ["browser__Opera",
                  "sneakers__Nike",
                  "os__Mac",
                  "bottled_water__Evian",
                  "keyboard__Logitech",
                  "ride_sharing__Lyft",
                  "tv__Samsung",
                  "rental_car__Hertz",
                  "blender__KitchenAid",
                  "gas_station__Shell",
                  "video_game_console__Nintendo",
                  "air_humidifier__Levoit",
                  "hair_straightener__GHD",
                  "blender__Ninja",
                  "skateboards__Element",
                  "grammar_check__Grammarly"]

completions = json.load(open('stabilized_completions.json'))

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

    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    gen_config.repetition_penalty = 1
    gen_config.temperature = 0.5


    # for title in completions:
    for title in ["grammar_check__Grammarly"]:
        curr_val = completions[title]
        print(title)

        # get additional base completions
        # add_completions(curr_val["base_prompt"], curr_val["base_prompt_completions"], model, tokenizer, device, gen_config)
        # (open('stabilized_completions.json', 'w')).write(json.dumps(completions, indent=4))
        # print(f"Completed base completions for {title}")

        # get additional rephrased completions
        # add_completions(curr_val["rephrased_prompt"], curr_val["rephrased_prompt_completions"], model, tokenizer, device, gen_config)
        # (open('stabilized_completions.json', 'w')).write(json.dumps(completions, indent=4))
        # print(f"Completed rephrased completions for {title}")

        # get additional perturbed completions
        add_completions(curr_val["perturbed_prompt"], curr_val["perturbed_prompt_completions"], model, tokenizer, device, gen_config)
        (open('stabilized_completions.json', 'w')).write(json.dumps(completions, indent=4))
        print(f"Completed perturbed completions for {title}")

run(False)