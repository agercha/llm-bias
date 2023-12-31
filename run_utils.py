import torch
import numpy as np
import gc
import torch.nn as nn
import random
from nltk.corpus import wordnet 
import copy

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)

def generate(model, tokenizer, input_ids, gen_config=None):

    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32
        
    input_ids = input_ids.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)
    
    print(output_ids)
    print(dir(output_ids))

    return output_ids[0]

def successful(gen_str, success_strs, fail_strs, show=True):
    jailbroken = False
    jailbroken_ex = None
    gen_str_unpunctuated = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), gen_str))
    gen_str_unpunctuated = gen_str_unpunctuated.upper()
    gen_arr = gen_str_unpunctuated.split()
    for prefix in fail_strs:
        if prefix.strip().upper() in gen_arr:
            jailbroken_ex = prefix
            jailbroken = True
    present = False
    present_ex = None
    for prefix in success_strs:
        if prefix.strip().upper() in gen_arr:
            present_ex = prefix
            present = True
    if show:
        print(f"\nCompletion: {gen_str}")
        print(f'Present: {present} {present_ex}| Jailbroken: {jailbroken} {jailbroken_ex}')
    return present and not jailbroken, present, jailbroken

def get_ids(tokenizer, vals, device = "cuda:0"):
    return torch.tensor(tokenizer(vals).input_ids).to(device)

def get_ids_with_slices(tokenizer, vals1, vals2, device = "cuda:0"):
    prompt = vals1 + vals2
    toks1 = tokenizer(vals1).input_ids
    slice1 = slice(0, len(toks1))
    toks2 = tokenizer(vals2).input_ids
    slice2 = slice(slice1.stop, len(toks2))

    return torch.tensor(tokenizer(prompt).input_ids).to(device), slice1, slice2

def get_gradients(model, tokenizer, base_strs, end_strs):
    
    all_ids, base_slice, end_slice = get_ids_with_slices(tokenizer, base_strs, end_strs)
    loss_slice = slice(end_slice.start - 1, end_slice.stop - 1)
    base_ids = all_ids[base_slice]
    end_ids = all_ids[end_slice]

    embed_weights = model.model.embed_tokens.weight
    
    one_hot = torch.zeros(
        base_ids.shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        base_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    embeds = model.model.embed_tokens(all_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            input_embeds, 
            embeds[:,base_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    loss = torch.nn.CrossEntropyLoss()(logits[0,loss_slice,:], end_ids)
    
    loss.backward()

    grad = one_hot.grad.clone()
    
    return grad / grad.norm(dim=-1, keepdim=True)

def get_cands(prompt, thesarus):
    prompt_words = prompt.split()
    count = 0
    for word in prompt_words:
        if word in thesarus: count += 1
    return count

def get_replacements(prompt, thesarus):
    # current_replacement
    prompt_words = prompt.split()
    all_prompts = [prompt_words]
    for ind, word in enumerate(prompt_words):
        if word in thesarus:
            syns = thesarus[word]
            all_prompts = [curr_prompt[:ind] + [syn] + curr_prompt[ind+1:] for syn in syns for curr_prompt in all_prompts]
    return [' '.join(curr_prompt) for curr_prompt in all_prompts]

def new_control(tokenizer, toks, grad, nonascii_toks, batch_size=8, topk=2500):
    grad[:, nonascii_toks.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices

    original_toks = toks.repeat(batch_size, 1)

    for i in range(batch_size):
        while True:
            old_ind = random.randint(0, len(toks) - 1)
            old_id = original_toks[i][old_ind]
            old_word_str = tokenizer.decode(old_id)
            new_id = top_indices[random.randint(0, top_indices.shape[0] - 1)][random.randint(0, topk - 1)]
            new_word_str = tokenizer.decode(new_id)
            old_wordsets = wordnet.synsets(old_word_str)
            new_wordsets = wordnet.synsets(new_word_str)
            if old_wordsets != [] and new_wordsets != [] and (old_word_str.strip().upper() != new_word_str.strip().upper()):
                best_old_wordset = old_wordsets[0]
                best_new_wordset = new_wordsets[0]
                sim = best_old_wordset.path_similarity(best_new_wordset)
                if sim != None and sim == 1 and new_id != old_id:
                    break

        original_toks[i][old_ind] = new_id

    return original_toks



def sample_control(toks, grad, nonascii_toks, batch_size=512, topk=2500):
    grad[:, nonascii_toks.to(grad.device)] = np.infty
    
    top_indices = (-grad).topk(topk, dim=1).indices

    toks = toks.to(grad.device)

    # repeat prompt batch size times = [[toks], [toks], .... , [toks]]
    original_toks = toks.repeat(batch_size, 1)

    # positions of [0, len(toks)/batch_size, 2*len(toks)/batch_size, ... , len(toks)]
    new_token_pos = torch.arange(
        0, 
        len(toks), 
        len(toks) / batch_size,
        device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )

    # replace values at new positions with new token vals
    new_control_toks = original_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks
    
def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
    return cands

def forward(*, model, input_ids, attention_mask, batch_size=8):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def try_loss(model, tokenizer, base_strs, end_strs, attention_mask, labels, decode_hidden_index = -1):
    base_ids = get_ids(tokenizer, base_strs)
    end_ids = get_ids(tokenizer, end_strs)
    output = model(input_ids=base_ids, attention_mask=attention_mask, labels = labels, output_hidden_states = True)
    loss, logits, hiddens = output['loss'], output['logits'], output['hidden_states']
    return loss, logits, hiddens[decode_hidden_index]

def get_loss(model, tokenizer, base_strs, end_strs, test_controls, batch_size=8):

    all_ids, base_slice, end_slice = get_ids_with_slices(tokenizer, base_strs, end_strs)
    base_ids = all_ids[base_slice]
    end_ids = all_ids[end_slice]

    max_len = base_slice.stop - base_slice.start
    test_ids = [
        get_ids(tokenizer, control)
        for control in test_controls
    ]
    pad_tok = 0
    max_len = max([test_ids1.shape[0] for test_ids1 in test_ids])
    while pad_tok in all_ids or any([pad_tok in ids for ids in test_ids]):
        pad_tok += 1
    nested_ids = torch.nested.nested_tensor(test_ids)
    test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))

    locs = torch.arange(base_slice.start, base_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        all_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    del locs, test_ids ; gc.collect()
    logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(end_slice.start-1, end_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,end_slice])
    return loss.mean(dim=-1)

# def get_loss2(model, tokenizer, base_str, end_str, test_controls):
#     base_ids = get_ids(tokenizer, base_str)
#     end_ids = get_ids(tokenizer, end_str)
#     logits = nn.Linear(base_ids.shape, end_ids.shape)(end_ids)
#     loss = nn.CrossEntropyLoss(logits, end_ids)
#     return loss

def my_loss(model, tokenizer, input_str, end_strs):
    input_ids = [
        torch.tensor(tokenizer(f"{input_str} {control}").input_ids).to("cuda:0")
        for control in end_strs
    ]
    l = torch.tensor(tokenizer(f"{input_str} ").input_ids).to("cuda:0").shape[0]
    pad_tok = 0
    max_len = max([test_ids1.shape[0] for test_ids1 in input_ids])
    while any([pad_tok in ids for ids in input_ids]):
        pad_tok += 1
    nested_ids = torch.nested.nested_tensor(input_ids)
    input_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(input_ids), max_len)).to("cuda:0")


    labels = input_ids.clone() 
    labels[:,:l] = -100

    res = model.forward(input_ids=input_ids,
                        labels=labels,
                        return_dict=True)

    return res.loss.item()