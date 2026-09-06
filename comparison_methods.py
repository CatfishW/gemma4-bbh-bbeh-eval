"""Bounded prompt strategies and local/API model backends.

No downloads at import or execution. Gold labels never enter method/backend APIs.
Published-paper claims and same-checkpoint transfer experiments are distinct.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from urllib import request, parse


from comparison_data import digest, vote_key


SUFFIXES = {
 'direct':'Return only the final answer in the format requested by the question.',
 'cot':'Think step by step. End with: Final answer: <answer>.',
 'cod':'Think step by step, but only keep minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.',
 'plan_solve':'First understand the problem and devise a plan. Then carry out the plan. End with: Final answer: <answer>.',
 'self_verify':'Solve the problem, check your answer against the conditions once, then return only the final answer.',
 'native_thinking':'Solve the problem. End with: Final answer: <answer>.',
}


@dataclass(frozen=True)
class Method:
    name: str = 'cot'
    budget: int = 512  # TOTAL completions, including critique and selection
    k: int = 3
    temperature: float = .7
    final_budget: int = 128
    rounds: int = 1
    demonstrations: str = ''

    def __post_init__(self):
        if self.name not in set(SUFFIXES)|{'self_consistency','self_rank','self_refine'}:
            raise ValueError('unsupported method; upstream modules use import-run, not renamed prompts')
        for key in ('budget','k','final_budget','rounds'):
            if type(getattr(self,key)) is not int or not 1<=getattr(self,key)<=131072:
                raise ValueError(f'invalid {key}')
        if not math.isfinite(self.temperature) or not 0<=self.temperature<=2: raise ValueError('invalid temperature')
        if self.k>32 or self.rounds>4: raise ValueError('branch/round limit')
        count = self.k + int(self.name=='self_rank') if self.name in {'self_rank','self_consistency'} else 1+2*self.rounds if self.name=='self_refine' else 1
        if self.budget<count: raise ValueError('budget smaller than required positive-cap calls')
        if self.name=='self_rank' and self.budget<=self.final_budget: raise ValueError('no candidate budget remains')


@dataclass
class Generation:
    text: str
    output_tokens: int | None
    input_tokens: int | None
    elapsed_seconds: float
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)
    error: str | None = None


class BackendResourceError(RuntimeError):
    """Infrastructure failure: abort and preserve the completed evaluation prefix."""


def resource_failure(exc):
    return isinstance(exc, MemoryError) or type(exc).__name__ in {'OutOfMemoryError'} or any(
        word in str(exc).lower() for word in ('cuda out of memory','cuda error:','device-side assert'))


def run_method(question: str, method: Method, backend, seed: int):
    """Backend and method see a QUESTION ONLY. No labels or correctness-triggered retry."""
    calls=[]; charged=0
    def ask(prompt, cap, temp):
        nonlocal charged
        cap=min(cap,method.budget-charged)
        if cap<1: raise RuntimeError('completion budget exhausted')
        started=time.perf_counter()
        try:
            out=backend.generate(prompt,cap,temp,int(digest([seed,len(calls),question])[:8],16)&0x7fffffff,
                                 thinking=method.name=='native_thinking')
        except Exception as exc:
            if resource_failure(exc):raise BackendResourceError(str(exc)) from exc
            out=Generation('',None,None,time.perf_counter()-started,error=type(exc).__name__)
        for n in (out.output_tokens,out.input_tokens):
            if n is not None and (type(n) is not int or n<0): raise ValueError('invalid backend usage')
        if out.output_tokens is not None and out.output_tokens>cap:
            raise ValueError('backend exceeded completion ceiling')
        charged += out.output_tokens if out.output_tokens is not None else cap
        calls.append({'prompt_sha256':digest(prompt),'requested_cap':cap,'generation':asdict(out)})
        if out.error: raise RuntimeError('generation failed; no retries; telemetry retained')
        return out.text
    prefix=(method.demonstrations+'\n\n') if method.demonstrations else ''
    prompt=prefix+question+'\n\n'+SUFFIXES.get(method.name,SUFFIXES['cot'])
    error=None; answer=''
    try:
        if method.name in SUFFIXES:
            answer=ask(prompt,method.budget,method.temperature)
        elif method.name in {'self_consistency','self_rank'}:
            reserve=method.final_budget if method.name=='self_rank' else 0
            cap=(method.budget-reserve)//method.k
            candidates=[ask(prompt,cap,method.temperature) for _ in range(method.k)]
            if method.name=='self_consistency':
                keys=[vote_key(c) for c in candidates]; counts=Counter(k for k in keys if k)
                winner=max(counts,key=lambda k:(counts[k],-keys.index(k))) if counts else ''
                answer=candidates[keys.index(winner)] if winner else ''
            else:
                selection=prefix+question+'\n\n'+ '\n\n'.join(f'Candidate {i+1}:\n{v}' for i,v in enumerate(candidates))
                answer=ask(selection+'\nCompare against the conditions. Return only the final answer.',reserve,0.0)
        else:
            cap=method.budget//(1+2*method.rounds)
            answer=ask(prompt,cap,method.temperature)
            for _ in range(method.rounds):
                critique=ask(question+'\nDraft:\n'+answer+'\nIdentify specific errors or say no error found.',cap,method.temperature)
                answer=ask(question+'\nDraft:\n'+answer+'\nFeedback:\n'+critique+'\nRevise only where justified. End with: Final answer: <answer>.',cap,method.temperature)
    except BackendResourceError:
        raise
    except RuntimeError as exc:
        error=str(exc);answer=''
    return {'prediction':answer,'calls':calls,'error':error,'charged_completion_tokens':charged,
            'output_tokens':sum(c['generation']['output_tokens'] for c in calls) if all(c['generation']['output_tokens'] is not None for c in calls) else None,
            'input_tokens':sum(c['generation']['input_tokens'] for c in calls) if all(c['generation']['input_tokens'] is not None for c in calls) else None}


class APIBackend:
    def __init__(self,url,model,timeout=120):
        parts=parse.urlsplit(url)
        if parts.scheme not in {'http','https'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('HTTP(S) endpoint without embedded secrets required')
        self.url,self.model,self.timeout=url.rstrip('/'),model,timeout

    def generate(self,prompt,cap,temperature,seed,thinking=False):
        payload={'model':self.model,'messages':[{'role':'user','content':prompt}],
                 'max_completion_tokens':cap,'temperature':temperature,'seed':seed,
                 'chat_template_kwargs':{'enable_thinking':thinking}}
        headers={'Content-Type':'application/json'}
        if os.getenv('OPENAI_API_KEY'): headers['Authorization']='Bearer '+os.environ['OPENAI_API_KEY']
        req=request.Request(self.url+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
        started=time.perf_counter()
        with request.urlopen(req,timeout=self.timeout) as resp: raw=resp.read(8_000_001)
        if len(raw)>8_000_000: raise ValueError('response size limit')
        value=json.loads(raw); choice=value['choices'][0];usage=value.get('usage',{})
        content=choice['message'].get('content') or ''
        if not isinstance(content,str): raise ValueError('text response required')
        return Generation(content,usage.get('completion_tokens'),usage.get('prompt_tokens'),
                          time.perf_counter()-started,choice.get('finish_reason'),
                          {'message':choice['message'],'usage':usage,'system_fingerprint':value.get('system_fingerprint')})


class LocalBackend:
    """Separate causal/multimodal loader; no CRAFT F/R-token requirement for baselines."""
    def __init__(self,path,device='cuda:0',adapter=None,max_context=16384,loader='legacy_gemma',profile=None,load_in_4bit=False,offload_kv_cache=False,max_gpu_memory_gib=8.,prefill_chunk_size=64):
        import torch
        if not Path(path).is_dir(): raise ValueError('local checkpoint path required')
        self.torch=torch; self.device=torch.device(device);self.max_context=max_context
        from comparison_runtime import limit_gpu_memory
        limit_gpu_memory(device,max_gpu_memory_gib)
        self.offload_kv_cache=offload_kv_cache
        self.prefill_chunk_size=prefill_chunk_size
        if offload_kv_cache:
            from rl_craft.attention import install_long_context_sdpa
            install_long_context_sdpa()
        if loader=='legacy_gemma':
            from rl.modeling import load_policy_model,load_tokenizer
            self.tokenizer=load_tokenizer(path);self.model=load_policy_model(path,device=device,load_in_4bit=load_in_4bit)
        elif loader=='causal':
            if load_in_4bit:raise ValueError('NF4 causal loader is not implemented; use an explicit supported runtime')
            from transformers import AutoModelForCausalLM,AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
            self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
                    torch_dtype=torch.bfloat16 if self.device.type=='cuda' else torch.float32).to(self.device)
        else: raise ValueError('unknown loader')
        if adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=False)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.profile=profile or {}
        self.end=self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.end=[self.end] if type(self.end)is int else list(self.end or [])
        if not self.end: raise ValueError('termination tokens required')
        self.pad=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.end[0]

    def generate(self,prompt,cap,temperature,seed,thinking=False):
        from transformers import GenerationConfig
        t=self.torch
        ids=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                add_generation_prompt=True,enable_thinking=thinking,**self.profile)
        if isinstance(ids,Mapping): ids=ids['input_ids']
        if not ids or any(type(i)is not int for i in ids): raise ValueError('invalid tokenized chat template')
        if len(ids)+cap>self.max_context: raise ValueError('context overflow; never silently truncate prompt')
        tokens=t.tensor([ids],device=self.device); mask=t.ones_like(tokens)
        config=GenerationConfig(do_sample=temperature>0,max_new_tokens=cap,top_k=0,top_p=1.,
                temperature=temperature if temperature>0 else 1.,pad_token_id=self.pad,eos_token_id=self.end,use_cache=True,cache_implementation='offloaded' if self.offload_kv_cache else None,
                prefill_chunk_size=self.prefill_chunk_size if self.offload_kv_cache else None)
        devices=list(range(t.cuda.device_count())) if t.cuda.is_available() else []
        if self.device.type=='cuda': t.cuda.synchronize(self.device)
        start=time.perf_counter()
        with t.random.fork_rng(devices=devices),t.inference_mode():
            t.manual_seed(seed);out=self.model.generate(input_ids=tokens,attention_mask=mask,generation_config=config)[0,len(ids):].tolist()
        if self.device.type=='cuda': t.cuda.synchronize(self.device)
        elapsed=time.perf_counter()-start; stopped=False
        for i,token in enumerate(out):
            if token in self.end: out=out[:i+1];stopped=True;break
        raw=self.tokenizer.decode(out,skip_special_tokens=False)
        error=None
        if callable(getattr(self.tokenizer,'parse_response',None)):
            try:
                parsed=self.tokenizer.parse_response(raw);content=parsed['content']
                if not isinstance(content,str): raise ValueError('invalid final channel')
            except (ValueError,KeyError,TypeError,IndexError):
                content='';error='channel_parse_failure'
        else:
            if thinking: raise ValueError('thinking requires an explicit channel parser; no heuristic mixing')
            content=self.tokenizer.decode(out,skip_special_tokens=True);parsed={'content':content}
        return Generation(content,len(out),len(ids),elapsed,'stop' if stopped else 'length',
                {'raw':raw,'channels':parsed if not error else {},'rendered_prompt_sha256':digest(ids),'token_ids':out},error)
