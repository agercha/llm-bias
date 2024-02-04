import json

# raw_thesarus = json.load(open('synonyms.json'))

# thesarus = dict()
# for word in raw_thesarus:
#     syns = raw_thesarus[word] + [word]
#     for syn in syns:
#         if syn not in thesarus:
#             thesarus[syn] = syns
#         else:
#             thesarus[syn] = list(set(syns + thesarus[syn]))
# print(thesarus)

# (open('thesarus1.json', 'w')).write(json.dumps(thesarus, indent=4))

# dataset = json.load(open('dataset.json'))

# for category in dataset:
#     for brand in dataset[category]["brands"]:
#         if brand not in dataset[category]["brands"][brand]:
#             dataset[category]["brands"][brand].append(brand)

# (open('dataset1.json', 'w')).write(json.dumps(dataset, indent=4))

# def single_successful(gen_str, target_strs):
#     gen_str_unpunctuated = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), gen_str))
#     gen_str_unpunctuated = gen_str_unpunctuated.upper()
#     # gen_arr = gen_str_unpunctuated.split()
#     present = False
#     for prefix in target_strs:
#         if prefix.strip().upper() in gen_str_unpunctuated:
#             present = True
#     return present

# dataset = json.load(open('dataset.json'))

# num_categories = len(dataset.keys())
# num_prompts = 0
# num_brands = 0

# for category in dataset:
#     num_prompts += len(dataset[category]["prompts"])
#     num_brands += len(dataset[category]["brands"])

# print(num_categories)
# print(num_prompts)
# print(num_brands)

# thesaurus = json.load(open('thesaurus.json'))

# seen_words = set()

# total_words = 0
# word_groups = 0
# for word in thesaurus:
#     total_words += 1
#     if word not in seen_words:
#         word_groups += 1
#         print(thesaurus[word])
#         for syn in thesaurus[word]:
#             seen_words.add(syn)

# print(total_words, word_groups)

def add_categories(seen_set, filename):
    completions = json.load(open(filename))
    for title in completions:
        seen_set.add(completions[title]["category"])

seen_categories = set()

add_categories(seen_categories, "completions.json")
# add_categories(seen_categories, "completions_temp_0_7.json")
add_categories(seen_categories, "completions_temp_0_5.json")

print(seen_categories)