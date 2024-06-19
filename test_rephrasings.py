import json
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GemmaForCausalLM, GemmaTokenizer)
from fastchat.model import get_conversation_template
from transformers import pipeline as transformer_pipeline
import torch

def generate(prompt, pipeline):

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
    outputs = pipeline(
        prompt,
        # max_new_tokens=64,
        eos_token_id=terminators,
        do_sample=True,
        # temperature=0.6,
        temperature=1,
        top_p=0.9,
    )
    return outputs[0]["generated_text"]

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
pipeline = transformer_pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

dataset = json.load(open("rephrasings_results.json"))

for category in dataset:
    for original_prompt in dataset[category]:

        original_prompt_completions = dataset[category][original_prompt]["original_prompt_completions"]
        rephrased_prompt = dataset[category][original_prompt]["rephrased_prompt"]
        rephrased_prompt_completions = dataset[category][original_prompt]["rephrased_prompt_completions"]

        while len(original_prompt_completions) < 100:
            response = generate(original_prompt, pipeline)
            original_prompt_completions.append(response)

        while len(rephrased_prompt_completions) < 100:
            response = generate(rephrased_prompt, pipeline)
            rephrased_prompt_completions.append(response)

        (open(f'rephrasings_results.json', 'w')).write(json.dumps(res, indent=4))
