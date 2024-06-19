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

dataset = json.load(open("dataset.json"))

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

res = {

}

for category in dataset:
    res[category] = {
    }

    for original_prompt in dataset[category]["prompts"]:

        new_prompt = f'Rephrase the prompt "{original_prompt}" in a way that would result in a very different answer. Respond only with the new prompt.'

        rephrased_prompt = generate(new_prompt, pipeline)

        res[category][original_prompt] = rephrased_prompt

        (open(f'rephrased_dataset.json', 'w')).write(json.dumps(res, indent=4))