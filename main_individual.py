import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM)
from fastchat.model import get_conversation_template
from run_utils import *

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

# model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#     ).to("cuda:0").eval()
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

tokenizer.pad_token_id = 0


def get_word(l):
    return l.split("\t")[0]

def get_syns(l):
    res = set(l.split("\t")[1].strip().split())
    res.add(l.split("\t")[0])
    return res

with open("word_docs/synonyms.txt", "r") as f:
    thesarus = {get_word(line):get_syns(line) for line in f.readlines()}

def do_run(curr_type,
           categories,
           test_size=1000,
           temp=0.5):
    
    np.random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)

    gen_config = model.generation_config
    gen_config.max_new_tokens = 256
    gen_config.repetition_penalty = 1
    gen_config.temperature = temp

    # categories_dict = dict()
    # for category in categories:
    #     categories_dict[category] = (open(f"word_docs/{curr_type}/{category}.txt", "r").readline()).split()

    prompts = open(f"word_docs/{curr_type}_prompts.txt", "r").readlines()

    success_dict = {category:0 for category in categories}

    for prompt in prompts:
        prompt_ids = get_ids(tokenizer, prompt)

        for _ in range(test_size):
            # run tests on all prompts
            completion = tokenizer.decode((generate(model, tokenizer, prompt_ids, gen_config=gen_config))).strip()
            print(completion)
            
            for category in categories:
                target_strs = [word.strip() for word in open(f"word_docs/{curr_type}/{category}.txt", "r").readlines()]
                print(target_strs)
                if single_successful(completion, target_strs): 
                    print("yes!")
                    success_dict[category] += 1
        
        for category in categories:
            # get loss
            target_strs = [word.strip() for word in open(f"word_docs/{curr_type}/{category}.txt", "r").readlines()]
            stripped_prompt = prompt.strip()
            loss =  my_loss(model, tokenizer, stripped_prompt, target_strs)
            # write loss
            with open(f"brand_results/{curr_type}_{category}_loss.txt", "a") as f:
                f.write(f"{loss}\n")

            # write scores using tests
            with open(f"brand_results/{curr_type}_{category}_score.txt", "a") as f:
                f.write(f"{success_dict[category] / test_size}\n")
            success_dict[category] = 0


do_run("streamingservice", ["Netflix", "Hulu", "Disney", "HBO", "Peacock", "Amazon"])

do_run("os", ["Mac", "Windows", "Linux"])

do_run("search", ["Bing", "DuckDuckGo", "Ecosia", "Google", "Yahoo"])

do_run("browser", ["Chrome", "Edge", "Firefox", "Opera", "Safari"])

do_run("chip", ["Intel", "Nvidia"])

do_run("llms", ["Vicuna", "Llama", "Claude", "ChatGPT", "Claude"])

do_run("phone", ["Apple", "Samsung", "Motorola", "Google", "Huaiwei"])