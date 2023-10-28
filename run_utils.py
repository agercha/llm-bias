import torch
import numpy as np

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
    
    return one_hot.grad.clone()

def sample_control(toks, grad, nonascii_toks, batch_size=256, topk=256, temp=1):
    grad[:, nonascii_toks.to(grad.device)] = np.infty
    
    top_indices = (-grad).topk(topk, dim=1).indices

    toks = toks.to(grad.device)

    original_toks = toks.repeat(batch_size, 1)
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
