import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

def single_successful(gen_str, target_strs):
    gen_str_unpunctuated = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), gen_str))
    gen_str_unpunctuated = gen_str_unpunctuated.upper()
    gen_arr = gen_str_unpunctuated.split()
    present = False
    for prefix in target_strs:
        if prefix.strip().upper() in gen_arr:
            present = True
    return present

dataset = json.load(open('dataset.json'))

completions = json.load(open('completions_temp_0_7.json'))

for title in list(reversed(completions.keys())):
    if title != "category_brand_promptindex":

        category = completions[title]["category"]
        brand = completions[title]["brand"]

        target_strs = dataset[category]["brands"][brand]
        all_brands = dataset[category]["brands"].keys()

        base_dict = {curr_brand:0 for curr_brand in all_brands}
        base_arr_dict = {curr_brand:[] for curr_brand in all_brands}
        sum_base = 0
        arr_base = []
        perturbed_dict = {curr_brand:0 for curr_brand in all_brands}
        perturbed_arr_dict = {curr_brand:[] for curr_brand in all_brands}
        sum_perturbed = 0
        arr_perturbed = []
        x = [i for i in range(len(completions[title]["base_prompt_completions"]))]

        # all_brands = [brand]

        fig = plt.figure(figsize=(14,6))
        max_y = 0
        colors = ["orangered", "gold", "cadetblue", "orchid", "yellowgreen", "plum", "slategrey", "maroon", "dodgerblue"]

        for brand_ind, curr_brand in enumerate(all_brands):
            curr_target_strs = dataset[category]["brands"][curr_brand]

            for i in range(len(completions[title]["base_prompt_completions"])):

                completion_base = completions[title]["base_prompt_completions"][i]
                success_base = single_successful(completion_base, curr_target_strs)
                if success_base: 
                    base_dict[curr_brand] += 1
                base_arr_dict[curr_brand].append(base_dict[curr_brand]/(i+1))

                completion_pertubed = completions[title]["perturbed_prompt_completions"][i]
                success_perturbed = single_successful(completion_pertubed, curr_target_strs)
                if success_perturbed: 
                    perturbed_dict[curr_brand] += 1
                perturbed_arr_dict[curr_brand].append(perturbed_dict[curr_brand]/(i+1))

            max_y = max(max_y, max(max(base_arr_dict[curr_brand][5:]), max(perturbed_arr_dict[curr_brand][5:])))

            plt.plot(x, base_arr_dict[curr_brand], label=f"base {curr_brand}", linestyle='dashed', color=colors[brand_ind])
            plt.plot(x, perturbed_arr_dict[curr_brand], label=f"perturbed {curr_brand}", color=colors[brand_ind])

        base_prompt = completions[title]["base_prompt"]
        perturbed_prompt = completions[title]["perturbed_prompt"]
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        filtered_category = category.replace("_", " ")
        plt.suptitle(f"Success for {filtered_category} prompt perturbed in the direction of {brand}", fontsize=12)
        plt.title(f"Base prompt: {base_prompt}\n\nPerturbed prompt (in direction of {brand}): {perturbed_prompt}", fontsize=6, wrap=True)
        plt.xlabel("Number of Completions")
        plt.ylabel("Average Score")
        plt.ylim(-0.01, max(0.1, max_y))
        
        plt.legend()
        plt.show()  
        # plt.savefig(f'graphs/{title}.png')