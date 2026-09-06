"""Avoid materializing repeated long KV tensors during single-token decoding.

Gemma's 512-dimensional global heads cannot use the CUDA Flash SDPA kernel in
this runtime. With no mask, Transformers enables GQA; its math fallback repeats
K/V across query heads and can allocate >1 GiB for a 96k prompt. A zero-stride
view expresses the same one-KV-head attention without this copy.
"""
import torch


def single_kv_decode(query, key, value, attention_mask=None, dropout=0., scaling=None):
    if query.shape[2]!=1 or key.shape[1]!=1 or value.shape[1]!=1:
        raise ValueError('single-token, single-KV-head attention required')
    key=key.expand(-1,query.shape[1],-1,-1)
    value=value.expand(-1,query.shape[1],-1,-1)
    result=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask,
                dropout_p=dropout,scale=scaling,is_causal=False,enable_gqa=False)
    return result.transpose(1,2).contiguous(),None


def install_long_context_sdpa():
    # Process-local registry only; installed solely in our offloaded inference
    # workers. The environment and other GPU service processes are untouched.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    if getattr(original,'_craft_long_context',False):return
    def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if (not torch.is_grad_enabled() and query.shape[2]==1 and key.shape[1]==1
                and key.shape[-1]>256 and key.shape[2]>4096):
            return single_kv_decode(query,key,value,attention_mask,dropout,scaling)
        return original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
    attention._craft_long_context=True
    ALL_ATTENTION_FUNCTIONS.register('sdpa',attention)
