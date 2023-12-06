# synonyms from https://github.com/xlhex/NLG_api_watermark/blob/main/meta_data/top800_syn_cand_adj.txt
import random

def get_word(l):
    return l.split("\t")[0]

def get_syns(l):
    return set(l.split("\t")[1].strip().split())

with open("word_docs/synonyms.txt", "r") as f:
    thesarus = {get_word(line):get_syns(line) for line in f.readlines()}

p = "I'm looking to create a scientist character with depth and authenticity. Can you provide an example that delves into not only their professional life but also their personal struggles, relationships, and the impact of their work on the broader world?"

def get_cands(prompt, thesarus):
    prompt_words = prompt.split()
    count = 0
    for word in prompt_words:
        if word in thesarus: 
            print(word)
            count += 1
    return count

def get_replacements(prompt, thesarus):
    # current_replacement
    prompt_words = prompt.split()
    all_prompts = [prompt_words]
    for ind, word in enumerate(prompt_words):
        if word in thesarus:
            syns = thesarus[word]
            all_prompts = [curr_prompt[:ind] + [syn] + curr_prompt[ind+1:] for syn in syns for curr_prompt in all_prompts]
    return [' '.join(curr_prompt) for curr_prompt in all_prompts]

print(get_replacements(p, thesarus))