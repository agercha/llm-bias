import torch
import numpy as np
from run_utils import *
import gc
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from fastchat.model import get_conversation_template
from numpy.linalg import norm

# https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
# https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html 

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

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

conv_template = get_conversation_template('llama-2')
conv_template.sep2 = conv_template.sep2.strip()

prompt = "I am an writer of young adult novels. I have a series of books I am working on and, in the newest book, and I have created a scientist character. Here is a short description this scientist: "

def get_ids(vals, device = "cuda:0"):
    # conv_template.append_message(conv_template.roles[0], vals)
    # prompt = conv_template.get_prompt()
    # conv_template.messages = []

    # return prompt
    return torch.tensor(tokenizer(vals).input_ids).to(device)

def get_embs(ids):
    return model.model.embed_tokens(ids)

cos0 = nn.CosineSimilarity(dim=0)
cos1 = nn.CosineSimilarity(dim=1)

# def cos_simm

all_id = get_ids(prompt)
all_emb = get_embs(all_id)
print(prompt)
print(all_id)
print(all_emb)

ind = 1

# https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/

for word in prompt.split():
    word_id = get_ids(word)
    l = len(word_id) - 1
    word_emb = get_embs(word_id)
    wordsets = wordnet.synsets(word)
    print(f"Word: {word}")
    print(f"ID: {word_id}")
    print(f"ID's reconstructed: {all_id[ind: ind+l]}")
    print(f"embedding: {word_emb}")
    ind += l
    if len(wordsets) > 1:
        syns = set(curr_syn.name() for curr_set in wordsets for curr_syn in curr_set.lemmas())

        print("Begin Synonyms:")
        f_all_syns = open(f"similarities/{word}_syn_allwords.txt", "w")
        # f_syn_words = open(f"similarities/{word}_syn_words.txt", "w")
        # f_syn_sim = open(f"similarities/{word}_syn_sims.txt", "w")
        for syn_str in syns:
            syn_id = get_ids(syn_str)
            # if len(syn_id) == len(word_id) and word != syn_str:
            if word != syn_str:
                # syn_emb = get_embs(syn_id)
                # sim1 = cos1(syn_emb[1:],word_emb[1:])[0]
                # print(f"Syn: {syn_str} \t\t| Similarity: {sim1}")
                # print(f"Syn Emb: \t\t{syn_emb[1:]}")
                # print(f"Original Emb: \t\t{word_emb[1:]}")
                f_all_syns.write(f"{syn_str}\n")
                # f_syn_sim.write(f"{sim1}\n")
        # f_syn_words.close()
        # f_syn_sim.close()
        f_all_syns.close()


        # ants = set(curr_ant.antonyms()[0].name() for curr_set in wordsets for curr_ant in curr_set.lemmas() if curr_ant.antonyms())
        # f_ant_sim = open(f"similarities/{word}_ant_sims.txt", "w")
        # f_ant_words = open(f"similarities/{word}_ant_words.txt", "w")
        # print("Begin Antonyms")
        # for ant_str in ants:
        #     ant_id = get_ids(ant_str)
        #     if len(ant_id) == len(word_id) and word != ant_str:
        #         ant_emb = get_embs(ant_id)
        #         sim1 = cos1(ant_emb[1:],word_emb[1:])[0]
        #         print(f"Ant: {ant_str} \t\t| Similarity: {sim1}")
        #         print(f"Syn Emb: \t\t{ant_emb[1:]}")
        #         print(f"Original Emb: \t\t{word_emb[1:]}")
        #         f_ant_words.write(f"{ant_str}\n")
        #         f_ant_sim.write(f"{sim1}\n")
        # f_ant_words.close()
        # f_ant_sim.close()

    print("\n\n\n")

print(prompt)
print(all_id)
print(all_emb)