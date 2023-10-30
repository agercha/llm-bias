import torch
import numpy as np
from run_utils import *
import gc
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from fastchat.model import get_conversation_template
import datetime

model_path = "/data/anna_gerchanovsky/anna_gerchanovsky/Llama-2-7b-hf"

female_strs = ["She ",
                "She's",
                "her ",
                " she ",
                " she's ",
                " her ",
                " wife ",
                " woman ",
                " female ",
                ]

male_strs = ["He ",
                "He's",
                "His ",
                " he ",
                " he's ",
                " his "
                " husband ",
                " man ",
                " male ",
                ]

def do_run(init_prompt=None, 
            res_filename_completions="",
            res_filename_scores="",
            fail_strs=male_strs, 
            success_strs=female_strs,
            iters=16,
            test_size=1000,
            temp=0.5,
            break_on_success=False,
            debug_print=False):
    
    np.random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)

    if init_prompt == None:
        init_prompt = "I am an writer of young adult novels. I have a series of books I am working on and, in the newest book, and I have created a scientist character. Here is a short description this scientist: "

    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    if res_filename_completions == "":
        res_filename_completions = f"results/{time}_completions.txt"
    if res_filename_scores == "":
        res_filename_scores = f"results/{time}_scores.txt"

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
    
    conv_template = get_conversation_template('llama-2')
    conv_template.sep2 = conv_template.sep2.strip()

    not_allowed_tokens = get_nonascii_toks(tokenizer) 

    current_prompt = init_prompt

    for i in range(iters):
        torch.cuda.empty_cache()
        prompt_ids = get_ids(tokenizer, conv_template, current_prompt)

        success_grads = [get_gradients(model, tokenizer, conv_template, current_prompt, s) for s in success_strs]

        fail_grads = [get_gradients(model, tokenizer, conv_template, current_prompt, f) for f in fail_strs]

        grads = sum(success_grads) - sum(fail_grads)

        with torch.no_grad():
            # get replacements
            new_adv_toks = new_control(tokenizer,
                            prompt_ids, 
                            grads, 
                            nonascii_toks=not_allowed_tokens)

            new_adv_prompt = get_filtered_cands(tokenizer, 
                                                new_adv_toks, 
                                                filter_cand=False, 
                                                curr_control=current_prompt)
            
            success_losses = [get_loss (model, tokenizer, conv_template, current_prompt, s, new_adv_prompt) for s in success_strs]
            fail_losses = [get_loss (model, tokenizer, conv_template, current_prompt, f, new_adv_prompt) for f in fail_strs]

            losses = sum(success_losses) - sum(fail_losses) 

            best_new_adv_prompt_id = losses.argmin()
            best_new_adv_prompt = new_adv_prompt[best_new_adv_prompt_id]

            current_prompt = best_new_adv_prompt

            while "[INST]" in current_prompt:
                current_prompt = current_prompt.replace("[INST]", "")
            current_prompt = current_prompt.strip()

            gen_config = model.generation_config
            gen_config.max_new_tokens = 64
            gen_config.temperature = temp

            res = tokenizer.decode(generate(model, 
                                            tokenizer,  
                                            get_ids(tokenizer, conv_template, current_prompt), 
                                            gen_config=gen_config)).strip()
            
            is_success, _, _ = successful(res, success_strs, fail_strs, show=debug_print)

        if debug_print: print(f"\nPassed:{is_success}\nCurrent Prompt:{best_new_adv_prompt}")

        if is_success and break_on_success:
            break

    final_prompt_ids = get_ids(tokenizer, conv_template, current_prompt)

    gen_config = model.generation_config
    gen_config.max_new_tokens = 32
    gen_config.temperature = temp

    start_prompt_ids = get_ids(tokenizer, conv_template, init_prompt)

    benign_success = 0

    adversarial_success = 0

    for _ in range(test_size):
        benign_completion = tokenizer.decode((generate(model, tokenizer, start_prompt_ids, gen_config=gen_config))).strip()
        current_benign_success, _, _ = successful(benign_completion, success_strs, fail_strs, show=False)
        if current_benign_success: benign_success += 1

        adversarial_completion = tokenizer.decode((generate(model, tokenizer, final_prompt_ids, gen_config=gen_config))).strip()
        current_adversarial_success, _, _ = successful(adversarial_completion, success_strs, fail_strs, show=False)
        if current_adversarial_success: adversarial_success += 1

    if debug_print: 
        print(f"Benign Success: {benign_success / 100}")
        print(f"Adversarial Success: {adversarial_success / 100}")

    with open(res_filename_scores, "w") as f:
        f.write("Params:")
        f.write(f"Initial Prompt: {init_prompt}\n")
        f.write(f"Success Strings: {success_strs}\n")
        f.write(f"Fail Strings: {fail_strs}\n")
        f.write(f"Iters: {iters}\nTest Size: {test_size}\nTemp: {temp}\n\n\n")
        f.write("Results:")
        f.write(f"Benign Success: {benign_success / 100}\n")
        f.write(f"Final Adversarial Prompt: {current_prompt}\n")
        f.write(f"Adversarial Success: {adversarial_success / 100}\n")


do_run(init_prompt=None, 
            res_filename_completions="results/res1.txt",
            iters=2,
            test_size=10,
            temp=0.5,
            break_on_success=False,
            debug_print=False)