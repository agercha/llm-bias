# synonyms from https://github.com/xlhex/NLG_api_watermark/blob/main/meta_data/top800_syn_cand_adj.txt

def get_word(l):
    return l.split("\t")[0]

def get_syns(l):
    return set(l.split("\t")[1].strip().split())

with open("synonyms.txt", "r") as f:
    thesarus = {get_word(line):get_syns(line) for line in f.readlines()}