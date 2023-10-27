import torch

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

def get_gradients(model, tokenizer, conv_template, base_strs, end_strs):
    
    all_ids = get_ids(tokenizer, conv_template, f"{base_strs}{end_strs}")
    base_slice = slice(0, len(base_strs))
    end_slice = slice(len(base_strs), len(all_ids))
    base_ids = all_ids[base_slice]
    end_ids = all_ids[end_slice]
    # all_ids = torch.cat((base_ids, end_ids))

    embed_weights = model.model.embed_tokens.weight

    # all_ids = torch.cat((base_ids, end_ids))
    
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
    
    end_embeds = model.model.embed_tokens(all_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            input_embeds, 
            end_embeds,
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    loss = torch.nn.CrossEntropyLoss()(logits[0,len(base_ids),:], end_ids)
    
    loss.backward()
    
    return one_hot.grad.clone()