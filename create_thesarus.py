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

dataset = json.load(open('dataset.json'))

for category in dataset:
    for brand in dataset[category]["brands"]:
        if brand not in dataset[category]["brands"][brand]:
            dataset[category]["brands"][brand].append(brand)

(open('dataset1.json', 'w')).write(json.dumps(dataset, indent=4))