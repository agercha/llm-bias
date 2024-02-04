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
completions = json.load(open('completions_temp_0_5.json'))
# completions = json.load(open('completions.json'))

# fig = plt.figure(figsize=(14,6))
# relative_loss_improvments = []
# absolute_score_improvements = []

rephrased = True

for title in list(reversed(completions.keys())):
    if title != "category_brand_promptindex" and title != "category_brand":

        category = completions[title]["category"]
        brand = completions[title]["brand"]

        target_strs = dataset[category]["brands"][brand]
        all_brands = list(dataset[category]["brands"].keys())
        all_brands.remove(brand)
        all_brands.append(brand) # brand of choice is last!

        base_dict = {curr_brand:0 for curr_brand in all_brands}
        base_arr_dict = {curr_brand:[] for curr_brand in all_brands}
        sum_base = 0
        arr_base = []

        perturbed_dict = {curr_brand:0 for curr_brand in all_brands}
        perturbed_arr_dict = {curr_brand:[] for curr_brand in all_brands}
        sum_perturbed = 0
        arr_perturbed = []

        rephrased_dict = {curr_brand:0 for curr_brand in all_brands}
        rephrased_arr_dict = {curr_brand:[] for curr_brand in all_brands}
        sum_rephrased = 0
        arr_rephrased = []

        x = [i for i in range(len(completions[title]["base_prompt_completions"]))]

        # all_brands = [brand]

        fig = plt.figure(figsize=(14,8))
        max_y = 0
        colors = ["gold", "cadetblue", "orchid", "yellowgreen", "blue", "slategrey", "maroon", "dodgerblue"]

        for brand_ind, curr_brand in enumerate(all_brands):
            curr_target_strs = dataset[category]["brands"][curr_brand]

            for i in range(len(completions[title]["base_prompt_completions"])):

                completion_base = completions[title]["base_prompt_completions"][i]
                success_base = single_successful(completion_base, curr_target_strs)
                if success_base: 
                    base_dict[curr_brand] += 1
                base_arr_dict[curr_brand].append(base_dict[curr_brand]/(i+1))

                completion_perturbed = completions[title]["perturbed_prompt_completions"][i]
                success_perturbed = single_successful(completion_perturbed, curr_target_strs)
                if success_perturbed: 
                    perturbed_dict[curr_brand] += 1
                perturbed_arr_dict[curr_brand].append(perturbed_dict[curr_brand]/(i+1))

                if rephrased:
                    completion_rephrased = completions[title]["rephrased_prompt_completions"][i]
                    success_rephrased = single_successful(completion_rephrased, curr_target_strs)
                    if success_rephrased: 
                        rephrased_dict[curr_brand] += 1
                    rephrased_arr_dict[curr_brand].append(rephrased_dict[curr_brand]/(i+1))

            max_y = max(max_y, max(max(base_arr_dict[curr_brand][5:]), max(perturbed_arr_dict[curr_brand][5:])))

            if rephrased: max_y = max(max_y, max(rephrased_arr_dict[curr_brand][5:]))

            if curr_brand == brand: curr_color = "orangered"
            else: curr_color = colors[brand_ind]

            # relative_loss_improvments.append(1 - (completions[title]["perturbed_prompt_loss"])/(completions[title]["base_prompt_loss"]))
            # absolute_score_improvements.append(perturbed_arr_dict[curr_brand][-1] - base_arr_dict[curr_brand][-1])

            plt.plot(x, base_arr_dict[curr_brand], label=f"base {curr_brand}", linestyle='dashed', color=curr_color)
            plt.plot(x, perturbed_arr_dict[curr_brand], label=f"perturbed {curr_brand}", color=curr_color)
            if rephrased: plt.plot(x, rephrased_arr_dict[curr_brand], label=f"rephrased {curr_brand}", color=curr_color, linestyle="dotted")

        base_prompt = completions[title]["base_prompt"]
        perturbed_prompt = completions[title]["perturbed_prompt"]
        if rephrased: rephrased_prompt = completions[title]["rephrased_prompt"]
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        filtered_category = category.replace("_", " ")
        plt.suptitle(f"Success for {filtered_category} prompt perturbed in the direction of {brand}", fontsize=12)
        if rephrased:
            plt.title(f"Base prompt: {base_prompt}\nPerturbed prompt (in direction of {brand}): {perturbed_prompt}\nRephrased prompt: {rephrased_prompt}", fontsize=6, wrap=True)
        else:
            plt.title(f"Base prompt: {base_prompt}\n\nPerturbed prompt (in direction of {brand}): {perturbed_prompt}", fontsize=6, wrap=True)
        plt.xlabel("Number of Completions")
        plt.ylabel("Average Score")
        plt.ylim(-0.01, max(0.1, max_y))
        
        plt.legend()
        # plt.show()  
        plt.savefig(f'graphs/05temp/{title}.png')
