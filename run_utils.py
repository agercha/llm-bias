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
    # input_ids = get_ids(tokenizer, conv_template, input_strs, device = "cuda:0")

    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    # if gen_config.max_new_tokens > 50:
    #     print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        
    input_ids = input_ids.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids

def successful(gen_str, success_strs, fail_strs):
    jailbroken = any([prefix in gen_str for prefix in fail_strs])
    present = any([prefix in gen_str for prefix in success_strs])
    print(f'\nPresent: {present} | Jailbroken: {jailbroken}')
    return present and not jailbroken

def get_ids(tokenizer, conv_template, vals, device = "cuda:0"):
    conv_template.append_message(conv_template.roles[0], vals)
    prompt = conv_template.get_prompt()
    conv_template.messages = []

    # return prompt
    return torch.tensor(tokenizer(prompt).input_ids).to(device)

def get_ids_with_slices(tokenizer, conv_template, vals1, vals2, device = "cuda:0"):
    conv_template.append_message(conv_template.roles[0], f"{vals1} {vals2}")
    prompt = conv_template.get_prompt()
    conv_template.messages = []

    conv_template.append_message(f"{vals1}", None)
    toks1 = tokenizer(conv_template.get_prompt()).input_ids
    slice1 = slice(0, len(toks1))

    conv_template.append_message(f"{vals2}", None)
    toks2 = tokenizer(conv_template.get_prompt()).input_ids
    slice2 = slice(slice1.stop, len(toks2))

    conv_template.messages = []

    # return prompt
    return torch.tensor(tokenizer(prompt).input_ids).to(device), slice1, slice2

def get_gradients(model, tokenizer, conv_template, base_strs, end_strs):
    
    all_ids, base_slice, end_slice = get_ids_with_slices(tokenizer, conv_template, base_strs, end_strs)
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

def get_replacements(tokenizer, conv_template, curr_prompt, batch_size=512, device="cuda:0"):
    prompt_words = curr_prompt.split()
    # perhaps use wup_similarity
    # original_words = prompt_words.repeat(batch_size, 1)
    new_set = []
    
    for i in range(batch_size):
        l = 0
        while l == 0:
            replace_ind = random.randint(0, len(prompt_words) - 1)
            old_word = prompt_words[replace_ind]
            # print(old_word)
            raw_arr = wordnet.synsets(old_word)
            arr = [word.name() for syn in raw_arr for word in syn.lemmas()]
            l = len(arr)
        new_word = random.choice(arr)
        print(old_word, new_word)
        copied_words = copy.deepcopy(prompt_words)
        copied_words[replace_ind] = new_word
        new_prompt = " ".join(copied_words)
        new_ids = get_ids(tokenizer, conv_template, new_prompt)
        # print(curr_prompt, new_prompt)
        new_set.append(new_prompt)


    return torch.Tensor(new_set).to(device)

def new_control(tokenizer, toks, grad, nonascii_toks, batch_size=512, topk=256):
    grad[:, nonascii_toks.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices

    original_toks = toks.repeat(batch_size, 1)

    for i in range(batch_size):
        # TODO FIX first index, no clue why it is this
        old_ind = random.randint(0, len(toks) - 1)
        old_word_str = tokenizer.decode(
        new_ind = top_indices[random.randint(0, top_indices.shape[0] - 1)][random.randint(0, topk - 1)]

        original_toks[i][old_ind] = new_ind

    return original_toks



def sample_control(toks, grad, nonascii_toks, batch_size=512, topk=256):
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

    # torch_randint = selects indices of random top grads
    # new_token_val[i] = top_indices [ top_grads[i] ]

    # rand_inds = torch.randint(0, topk, (batch_size, 1), device=grad.device)
    # new_val_indices = top_indices[rand_inds]
    # print(new_val_indices)

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

def forward(*, model, input_ids, attention_mask, batch_size=512):

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

def get_loss(model, tokenizer, conv_template, base_strs, end_strs, test_controls, batch_size=512):
    # control slice? the prompt slice
    # test control = new_adv_prompt
    # return ids TRUE

    all_ids, base_slice, end_slice = get_ids_with_slices(tokenizer, conv_template, base_strs, end_strs)
    base_ids = all_ids[base_slice]
    end_ids = all_ids[end_slice]

    max_len = base_slice.stop - base_slice.start
    test_ids = [
        torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
        for control in test_controls
    ]
    pad_tok = 0
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