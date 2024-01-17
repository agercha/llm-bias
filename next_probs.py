# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
import torch
from pprint import pprint
from transformers import (AutoModelForCausalLM, AutoTokenizer)

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda:0").eval()

tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch


tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

input_texts = ["What is the best streaming service? Hulu.", "What is the best streaming service? Netflix."]

batch = to_tokens_and_logprobs(model, tokenizer, input_texts)
pprint(batch)