import sys
import json
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from transformers import pipeline as transformer_pipeline

category = sys.argv[1]
index = sys.argv[2]

filename = f"{category}_{index}.json"

dataset = json.load(open('dataset.json'))

prompt = dataset[category]['prompts'][int(index)]

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/meta-llama/Meta-Llama-3-8B-Instruct"

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
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    responses.append(outputs[0]["generated_text"][len(prompt):])

    if i%10 == 0:
        (open(filename, 'w')).write(json.dumps(responses, indent=4))