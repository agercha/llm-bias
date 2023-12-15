import torch
import numpy as np
from run_utils import *
import gc
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from fastchat.model import get_conversation_template
import datetime

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

female_strs = (open("word_docs/female.txt").readline()).split()
male_strs = (open("word_docs/male.txt").readline()).split()

conservative_strs = (open("word_docs/conservative.txt").readline()).split()
liberal_strs = (open("word_docs/liberal.txt").readline()).split()

hulu_strs = ["Hulu", "hulu", "HULU"]
netflix_strs = ["Netflix", "netflix", "NETFLIX"]

bud_strs = ["bud", "budweiser", "Bud"]
coors_strs = ["coors", "Coors", "COORS"]

mcdonalds_strs = ["mcdonalds", "McDonalds", "MCDONALDS", "bigmac", "McNuggets", "mcnuggets", "BigMac", "BIGMAC"]
burgerking_strs = ["King", "BurgerKing", "whopper"]

BMW_strs = ["BMW"]
Mercedes_strs = ["Mercedes"]

snickers_strs = ["snickers"]
twix_strs = ["twix"]

model = AutoModelForCausalLM.from_pretrained(
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

conv_template = get_conversation_template('llama-2')
conv_template.sep2 = conv_template.sep2.strip()

def test_all(init_prompts_filename,
            # res_filename_completions="",
            # res_filename_scores="",
            all_prompts_filename="",
            all_success_filename="",
            all_fail_filename="",
            fail_strs=male_strs, 
            success_strs=female_strs,
            test_size=1000,
            temp=0.5):
    
    np.random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)

    # if init_prompt == None:
    #     init_prompt = "I am an writer of young adult novels. I have a series of books I am working on and, in the newest book, and I have created a scientist character. Here is a short description this scientist: "

    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # if res_filename_completions == "":
    #     res_filename_completions = f"results/{time}_completions.txt"
    # if res_filename_scores == "":
    #     res_filename_scores = f"results/{time}_scores.txt"
    if all_prompts_filename == "":
        all_prompts_filename = f"results/{time}_prompts.txt"
    if all_success_filename == "":
        all_success_filename = f"results/{time}_successs.txt"
    if all_fail_filename == "":
        all_fail_filename = f"results/{time}_fail.txt"

    # current_prompt = init_prompt
    all_prompts = open(init_prompts_filename, "r").readlines()

    gen_config = model.generation_config
    gen_config.max_new_tokens = 32
    gen_config.repetition_penalty = 1
    gen_config.temperature = temp


    # iters
    # we may not want to run a loop, rather adjust current replacement so it does all possibilities
    torch.cuda.empty_cache()

    # old_replacement
    # uncomment this section to use old method 

    with torch.no_grad():
        # get replacements

        # current_replacement
        # new_adv_prompt = get_replacements(current_prompt, thesarus)

        adversarial_success = 0
        success_present = 0
        fail_present = 0

        for curr_adv_prompt_i in all_prompts:
            adversarial_success = 0
            success_present = 0
            fail_present = 0
            for _ in range(test_size):
                curr_adv_prompt_ids = get_ids(tokenizer, curr_adv_prompt_i)
                adversarial_completion = tokenizer.decode((generate(model, tokenizer, curr_adv_prompt_ids, gen_config=gen_config))).strip()
                current_adversarial_success, current_success_present, current_fail_present = successful(adversarial_completion, success_strs, fail_strs, show=False)
                if current_adversarial_success: adversarial_success += 1
                if current_success_present: success_present+= 1
                if current_fail_present: fail_present += 1

            with open(all_prompts_filename, "a") as f:
                f.write(f"{curr_adv_prompt_i}\n")
            with open(all_success_filename, "a") as f:
                f.write(f"{success_present/test_size}\n")
            with open(all_fail_filename, "a") as f:
                f.write(f"{fail_present/test_size}\n")


test_all(init_prompts_filename="word_docs/doctor_prompts.txt",
            all_prompts_filename="search_results/doctor.txt",
            all_success_filename="search_results/doctor_female_scores.txt",
            all_fail_filename="search_results/doctor_male_scores.txt",
            fail_strs=male_strs, 
            success_strs=female_strs,
            test_size=1000,
            temp=0.5)


test_all(init_prompts_filename="word_docs/scientist_prompts.txt",
            all_prompts_filename="search_results/scientist.txt",
            all_success_filename="search_results/scientist_female_scores.txt",
            all_fail_filename="search_results/scientist_male_scores.txt",
            fail_strs=male_strs, 
            success_strs=female_strs,
            test_size=1000,
            temp=0.5)


test_all(init_prompts_filename="word_docs/burger_prompts.txt",
            all_prompts_filename="search_results/burger.txt",
            all_success_filename="search_results/burger_mcdonalds_scores.txt",
            all_fail_filename="search_results/burger_burgerking_scores.txt",
            fail_strs=burgerking_strs, 
            success_strs=mcdonalds_strs,
            test_size=1000,
            temp=0.5)


test_all(init_prompts_filename="word_docs/beers_prompts.txt",
            all_prompts_filename="search_results/beers.txt",
            all_success_filename="search_results/beers_coors_scores.txt",
            all_fail_filename="search_results/beers_bud_scores.txt",
            fail_strs=bud_strs, 
            success_strs=coors_strs,
            test_size=1000,
            temp=0.5)


test_all(init_prompts_filename="word_docs/streamingservice_prompts.txt",
            all_prompts_filename="search_results/streamingservice.txt",
            all_success_filename="search_results/streamingservice_netflix_scores.txt",
            all_fail_filename="search_results/streamingservice_hulu_scores.txt",
            fail_strs=hulu_strs, 
            success_strs=netflix_strs,
            test_size=1000,
            temp=0.5)