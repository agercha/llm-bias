import os
import json
import matplotlib.pyplot as plt
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from fastchat.model import get_conversation_template
from transformers import pipeline as transformer_pipeline
from matplotlib.ticker import PercentFormatter
import torch

def my_loss(model, tokenizer, input_str, end_strs, device):
    messages = [
        {"role": "user", "content":input_str},
    ]
    input_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    input_ids = [torch.tensor(tokenizer(f"{input_str}{end_str}").input_ids).to(device) for end_str in end_strs] 
    l = torch.tensor(tokenizer(f"{input_str}").input_ids).to(device).shape[0] - 1
    nested_ids = torch.nested.nested_tensor(input_ids)


    pad_tok = 0
    max_len = max([test_ids1.shape[0] for test_ids1 in input_ids])
    while any([pad_tok in ids for ids in input_ids]):
        pad_tok += 1
    input_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(input_ids), max_len)).to(device)

    attention_mask = torch.ones(input_ids.shape).to(device)
    attention_mask[input_ids == 0] = 0

    labels = input_ids.clone().to(device)
    labels[:, :l] = -100

    res = model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()

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

entries_long = os.listdir('long_completions_gemma7bit_temp1/')

device = "cuda:0"
# device = "cpu"

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/gemma-7b-it"
# model_path = "/Users/annagerchanovsky/Desktop/classrn/gemma-2b-it/"
model = GemmaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipeline = transformer_pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

scores1 = []
scores2 = []

losses1 = []
losses2 = []

for entry in entries_long:
    print(entry)
    res = json.load(open(f'long_completions_gemma7bit_temp1/{entry}'))
    category = entry.split(".")[0]

    all_brands = list(dataset[category]["brands"].keys())

    # arr_dict = {curr_brand:[] for curr_brand in all_brands}

    for ind in res:
        if (len(res[ind]["base_prompt_completions"]) == 1000): # what we want
            completions1 = res[ind]["base_prompt_completions"][:500]
            completions2 = res[ind]["base_prompt_completions"][500:]

            for brand in all_brands:
                curr_target_strs = dataset[category]["brands"][brand]

                scores1.append(sum([single_successful(curr_res, curr_target_strs) for curr_res in completions1])/500)
                scores2.append(sum([single_successful(curr_res, curr_target_strs) for curr_res in completions2])/500)

                losses1.append(sum([my_loss(model, tokenizer, curr_res, curr_target_strs, device) for curr_res in completions1])/500)
                losses2.append(sum([my_loss(model, tokenizer, curr_res, curr_target_strs, device) for curr_res in completions2])/500)

                print(scores1)
                print(scores2)
                print(losses1)
                print(losses2)
                assert(False)


fig = plt.figure(figsize=(8,8))

plt.suptitle("Scores of all brands on split completions")
plt.title("Scores are calculated over 500 completions", fontsize=8)

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Score on first half of completions")
plt.ylabel("Score on second half of completions")
plt.scatter(scores1, scores2, s=5)
# plt.show()
plt.savefig("a.png")

res_json = {
    "scores1": scores1,
    "scores2": scores2
}

open(f'halved_scores.json', "w").write(json.dumps(res_json, indent=4))
