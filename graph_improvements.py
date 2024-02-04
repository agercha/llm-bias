import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

def single_successful(gen_str, target_strs):
    gen_str_unpunctuated = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), gen_str))
    gen_str_unpunctuated = gen_str_unpunctuated.upper()
    # gen_arr = gen_str_unpunctuated.split()
    present = False
    for prefix in target_strs:
        if prefix.strip().upper() in gen_str_unpunctuated:
            present = True
    return present

dataset = json.load(open('dataset.json'))

# completions = json.load(open('completions_temp_0_7.json'))
# completions = json.load(open('completions.json'))

fig = plt.figure(figsize=(14,6))
relative_loss_improvments = []
absolute_score_improvements = []
relative_score_improvements = []


def add_scores(filename):
    completions = json.load(open(filename))
    for title in list(reversed(completions.keys())):
        if title != "category_brand_promptindex" and title != "category_brand":

            category = completions[title]["category"]
            brand = completions[title]["brand"]

            all_brands = list(dataset[category]["brands"].keys())
            all_brands.remove(brand)
            all_brands.append(brand) # brand of choice is last!

            base_dict = {curr_brand:0 for curr_brand in all_brands}

            perturbed_dict = {curr_brand:0 for curr_brand in all_brands}

            x = [i for i in range(len(completions[title]["base_prompt_completions"]))]

            all_brands = [brand]

            for brand_ind, curr_brand in enumerate(all_brands):
                curr_target_strs = dataset[category]["brands"][curr_brand]

                for i in range(len(completions[title]["base_prompt_completions"])):

                    completion_base = completions[title]["base_prompt_completions"][i]
                    success_base = single_successful(completion_base, curr_target_strs)
                    if success_base: 
                        base_dict[curr_brand] += 1

                    completion_perturbed = completions[title]["perturbed_prompt_completions"][i]
                    success_perturbed = single_successful(completion_perturbed, curr_target_strs)
                    if success_perturbed: 
                        perturbed_dict[curr_brand] += 1

                relative_loss_improvments.append(1 - (completions[title]["perturbed_prompt_loss"])/(completions[title]["base_prompt_loss"]))
                # if base_dict[curr_brand] == 0: relative_score_improvements.append(1)
                # else: relative_score_improvements.append(1 - perturbed_dict[curr_brand]/base_dict[curr_brand])
                absolute_score_improvements.append((perturbed_dict[curr_brand] - base_dict[curr_brand])/len(completions[title]["base_prompt_completions"]))

add_scores('completions.json')         
add_scores('completions_temp_0_5.json')          

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

plt.axhline(y = 0, color="black", linewidth=0.75)
plt.scatter(relative_loss_improvments, absolute_score_improvements)
# plt.scatter(relative_loss_improvments, relative_score_improvements)
plt.xlabel("Relative Loss Improvement")
# plt.ylabel("Relative Score Improvement")
plt.ylabel("Absolute Score Improvement")
# plt.show()
plt.savefig(f'graphs/score_improvment.png')