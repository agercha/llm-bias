import sys
import json
import torch
import os
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline
import matplotlib.pyplot as plt



category = sys.argv[1]
index = sys.argv[2]

filename = f"llama3it_test/{category}_{index}.json"

dataset = json.load(open('dataset.json'))

prompt = dataset[category]['prompts'][int(index)]

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B-Instruct"

pipeline = transformer_pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "user", "content": prompt},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

responses = []

for i in range(1000):
    outputs = pipeline(
        prompt,
        max_new_tokens=100000,
        eos_token_id=terminators,
        do_sample=True,
        # temperature=0.6,
        top_p=0.9,
    )

    responses.append(outputs[0]["generated_text"][len(prompt):])
    print(responses[-1])

    # if i%10 == 0:
    (open(filename, 'w')).write(json.dumps(responses, indent=4))
'''

entries = os.listdir("llama3it_test")
entries.remove("blank.txt")

str_lens = []
word_lens = []

for entry in entries:
    completions = json.load(open(f'llama3it_test/{entry}'))

    for completion in completions:
        str_lens.append(len(completion))
        word_lens.append(len(completion.split()))

        
fig = plt.figure(figsize=(14,8))
plt.suptitle("Llama3it lengths")
plt.xlabel("Number of characters")
plt.ylabel("Number of completions")
plt.hist(str_lens, bins=32)
plt.show()
        
fig = plt.figure(figsize=(14,8))
plt.suptitle("Llama3it lengths")
plt.xlabel("Number of words")
plt.ylabel("Number of completions")
plt.hist(word_lens, bins=32)
plt.show()
'''