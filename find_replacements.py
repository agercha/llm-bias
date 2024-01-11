def get_word(l):
    return l.split("\t")[0]

def get_syns(l):
    res = set(l.split("\t")[1].strip().split())
    res.add(l.split("\t")[0])
    return res

with open("word_docs/synonyms.txt", "r") as f:
    thesarus = {get_word(line):get_syns(line) for line in f.readlines()}

def get_cands(prompt, thesarus, banned=[]):
    prompt_words = prompt.split()
    count = 0
    options = 1
    for word in prompt_words:
        if word in thesarus and word not in banned: 
            count += 1
            options *= len(thesarus[word])
    # return count
    return options

for w in ["browser", "chip", "llms", "os", "phone", "search", "streamingservice"]:
    with open(f"word_docs/{w}_prompts.txt", "r") as f:
        for line in f.readlines():
            print(get_cands(line, thesarus, ["large"]))