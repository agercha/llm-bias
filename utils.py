from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, 
                          BertModel, BertTokenizer, TFBertModel, FlaxBertForCausalLM)
import torch.nn as nn
import numpy as np
import gc

def load_conversation_template(template_name):
    conv_template = get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer


class PromptManager:
    def __init__(self, *, tokenizer, conv_template, success_targets, fail_targets, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        # self.instruction = instruction
        self.success_targets = success_targets
        self.fail_targets = fail_targets
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
        # self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        # encoding = self.tokenizer(prompt)
        # toks = encoding.input_ids

        # self.conv_template.messages = []

        # self.conv_template.append_message(self.conv_template.roles[0], None)
        # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # self._user_role_slice = slice(None, len(toks))

        # self.conv_template.update_last_message(f"{self.adv_string}")
        # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

        # # self.conv_template.update_last_message(f"{self.adv_string}")
        # # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # self._control_slice = slice(self._goal_slice.stop, len(toks))

        # self.conv_template.append_message(self.conv_template.roles[1], None)
        # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

        # self.conv_template.update_last_message(f"{self.target}")
        # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
        # self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

        # self.conv_template.messages = []

        return prompt
    
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        # input_ids = torch.tensor(toks[:self._target_slice.stop])
        input_ids = torch.tensor(toks)

        return input_ids
    
    def get_any_ids(self, vals, device):
        return  [torch.tensor(self.tokenizer(val).input_ids).to(device) for val in vals]


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


def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
def get_embedding_matrix(model):
    return model.model.embed_tokens.weight

# def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
def token_gradients(model, input_ids, success_ids, fail_ids):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """


    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids.shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    # input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    overall_grad = None
    for s in success_ids:
        full_in = torch.cat((input_ids, s))
        embeds = get_embeddings(model, full_in.unsqueeze(0)).detach()
        # s_one_hot = torch.zeros(
        #     s.shape[0],
        #     embed_weights.shape[0],
        #     device=model.device,
        #     dtype=embed_weights.dtype
        # )
        # s_one_hot.scatter_(
        #     1, 
        #     s.unsqueeze(1),
        #     torch.ones(s_one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        # )
        # s_one_hot.requires_grad_()
        # s_embeds = (s_one_hot @ embed_weights).unsqueeze(0)
        # print(input_ids.unsqueeze(0))
        # print(type(s))
        # print(s)
        # embeds = get_embeddings(model, (torch.cat((input_ids, s)).unsqueeze(0))).detach()
        # # embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        # full_embeds = torch.cat(
        #     [
        #         embeds[:,:0,:], 
        #         input_embeds, 
        #         embeds[:,len(input_ids):,:]
        #     ], 
        #     dim=1)
        # full_embeds = input_embeds
        
        logits = model(inputs_embeds=embeds).logits
        targets = s[:]
        print(logits, s)
        loss = nn.CrossEntropyLoss()(logits[0,:,:], targets)
        
        loss.backward()
        
        grad = one_hot.grad.clone()
        if overall_grad == None:
            overall_grad = grad / grad.norm(dim=-1, keepdim=True)
        else:
            overall_grad += grad / grad.norm(dim=-1, keepdim=True)


    for f in fail_ids:
        embeds = get_embeddings(model, (input_ids + f).unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:,:0,:], 
                input_embeds, 
                embeds[:,len(input_ids):,:]
            ], 
            dim=1)
        # full_embeds = input_embeds
        
        logits = model(inputs_embeds=full_embeds).logits
        targets = f
        loss = nn.CrossEntropyLoss()(logits, targets)
        
        loss.backward()
        
        grad = one_hot.grad.clone()
        overall_grad -= grad / grad.norm(dim=-1, keepdim=True)
    
    return grad


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    # TODO - make similar

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)

    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        # step=1,
        device=grad.device
    ).type(torch.int64)
    # print(new_token_pos)
    # new_token_pos = torch.Tensor([1]).type(torch.int64).to(grad.device)
    # print(top_indices)
    # print(new_token_pos)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

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
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
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

def get_logits(*, model, tokenizer, input_ids, test_controls=None, return_ids=False, batch_size=512):
    
    if isinstance(test_controls[0], str):
        # max_len = control_slice.stop - control_slice.start
        # max_len = 
        test_ids = [
            # torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids, device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), len(input_ids)))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    # if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
    if not(test_ids[0].shape[0] == len(input_ids)):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {len(input_ids)}), " 
            f"got {test_ids.shape}"
        ))

    # locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    locs = torch.arange(0, len(input_ids)).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def target_loss_old(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    # loss = crit(logits.transpose(1,2), ids)
    return loss.mean(dim=-1)

def target_loss(logits, success_strs, fail_strs):
    crit = nn.CrossEntropyLoss(reduction='none')
    positive_loss = crit(logits.transpose(1,2), success_strs)
    negative_loss = crit(logits.transpose(1,2), fail_strs)
    return positive_loss - negative_loss

def get_losses(model, tokenizer, input_ids, test_controls, success_strs, fail_strs):

    s_loss = None
    f_loss = None

    for s in success_strs:
        # for control in test_controls:
        #     print(control + s)
        sucess_test_ids = [
                    # torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
                    torch.tensor(tokenizer(control + s, add_special_tokens=False).input_ids, device=model.device)
                    for control in test_controls
                ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in sucess_test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(sucess_test_ids)
        sucess_test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(sucess_test_ids), len(sucess_test_ids)))
        # sucess_test_ids = torch.nested.to_tensor(sucess_test_ids)
        locs = torch.arange(0, len(input_ids)).repeat(sucess_test_ids.shape[0], 1).to(model.device)
        ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(sucess_test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            sucess_test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None

        logits, ids = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=36), ids
        curr_loss = target_loss_old(logits, ids, slice(len(test_controls[0]), len(test_controls[0]) + len(s), None))
        if s_loss is None: s_loss = curr_loss
        else: s_loss += curr_loss

    for f in fail_strs:
        fail_test_ids = [
                    # torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
                    torch.tensor(tokenizer(control + f, add_special_tokens=False).input_ids, device=model.device)
                    for control in test_controls
                ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in fail_test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(fail_test_ids)
        fail_test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(fail_test_ids), len(fail_test_ids)))
        locs = torch.arange(0, len(input_ids)).repeat(fail_test_ids.shape[0], 1).to(model.device)
        ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(fail_test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            fail_test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None

        logits, ids = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=36), ids
        curr_loss = target_loss_old(logits, ids, slice(len(test_controls[0]), len(test_controls[0]) + len(f), None))
        if f_loss is None: f_loss = curr_loss
        else: f_loss += curr_loss

    return s_loss - f_loss



