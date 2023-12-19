def get_associations(category, category1, category2):
    d1 = dict()
    d2 = dict()

    prompts = open(f"search_results/{category}.txt", "r").readlines()
    scores1 = open(f"search_results/{category}_{category1}_scores.txt", "r").readlines()
    scores2 = open(f"search_results/{category}_{category2}_scores.txt", "r").readlines()

    for i in range(len(prompts)):
        for word in prompts[i].split():
            if word in d1: 
                d1[word].append(float(scores1[i]))
                d2[word].append(float(scores2[i]))
            else:
                d1[word] = [float(scores1[i])]
                d2[word] = [float(scores2[i])]

    with open(f"associations/{category}.txt", "w") as f:
        f.write(f"word\t{category1}\t{category2}\n")
        for key in d1.keys():
            f.write(f"{key}\t{sum(d1[key]) / len(d1[key])}\t{sum(d2[key]) / len(d2[key])}\n")


get_associations("doctor", "female", "male")
get_associations("scientist", "female", "male")
get_associations("burger", "mcdonalds", "burgerking")
get_associations("beers", "coors", "bud")
get_associations("streamingservice", "hulu", "netflix")