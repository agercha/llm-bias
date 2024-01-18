# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer)

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"
# model_path = "/Users/annagerchanovsky/Documents/Documents/research/Llama-2-7b-hf"

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


def to_tokens_and_logprobs(model, tokenizer, prompts, end):
    end_len = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to("cuda:0").shape[-1] - 1

    input_texts = [prompt + " " + end for prompt in prompts]
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to("cuda:0")
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    print(end_len)
    return(gen_probs)

    # batch = []
    # for input_sentence, input_probs in zip(input_ids, gen_probs):
    #     text_sequence = []
    #     for token, p in zip(input_sentence, input_probs):
    #         if token not in tokenizer.all_special_ids:
    #             text_sequence.append((tokenizer.decode(token), p.item()))
    #     batch.append(text_sequence)
    # return batch


# tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

# prompts = [
#     "I'm curious to know your thoughts on the top computer chip developer in the industry. Can you provide insights into the leading companies that excel in designing and producing high-quality computer chips?", 
#     "As I explore the world of computer chip development, I'm interested in discovering the foremost companies in this field. Could you share your expertise on which organizations are recognized for being the best computer chip developers, considering factors like innovation and performance?", 
#     "I'm looking to understand more about the premier computer chip developers in the market. Could you offer information on the companies that are widely regarded as leaders in creating cutting-edge computer chips, with a focus on factors such as technological advancements and reliability?", 
#     "In the realm of computer chip development, I'm seeking recommendations for the most reputable companies. Can you shed light on the top players in this industry, highlighting their achievements in innovation, performance, and overall chip quality?", 
#     "As I delve into the world of computer chip technology, I'm eager to know your opinion on the best developers in the field. Could you provide insights into the companies that are considered pioneers, taking into account their contributions to advancements, performance, and reliability in computer chip development?"
# ]

# input_texts = [input_text + " Nvidia" for input_text in prompts]

prompts = open(f"word_docs/browser_prompts.txt", "r").readlines()
ends = ["Netflix", "Hulu", "Disney", "HBO", "Peacock", "Amazon"]
for end in ends:
    # input_texts = [prompt + " " + end for prompt in prompts]
    batch = to_tokens_and_logprobs(model, tokenizer, prompts, end)

    for b in batch:
        print(b)
        assert(False)