"""Quality-first LoRA controls and a CRAFT-Q research arm.

Single-update, fresh-policy reference RLOO/GRPO/Dr.GRPO controls are explicitly
NOT full upstream PPO/DeGRPO/ThinkPrune reproductions. No silent label harvesting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import statistics
import time
import os

from comparison_suite import Case, Scorer, SUFFIXES, digest, environment, file_hash, load_cases, write_json, source_hashes


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = 'rloo'
    steps: int = 100
    roots: int = 4
    k: int = 4
    max_tokens: int = 512
    max_context: int = 16384
    rank: int = 32
    alpha: int = 64
    learning_rate: float = 2e-5
    temperature: float = .8
    seed: int = 7
    prompt: str = 'cot'
    max_sampled_tokens: int = 200000
    cost_weight: float = .1
    quality_warmup: int = 50
    cost_ramp: int = 50
    prefix_tokens: int = 64
    continue_tokens: int = 256
    answer_tokens: int = 128
    checkpoint_every: int = 25
    load_in_4bit: bool = False
    activation_offload: bool = False
    offload_kv_cache: bool = False
    max_gpu_memory_gib: float = 8.0

    def __post_init__(self):
        if self.algorithm not in {'rloo','grpo_reference','drgrpo_reference','sft_answers','craft_quality'}:
            raise ValueError('unknown training algorithm')
        for key in ('steps','roots','k','max_tokens','max_context','rank','alpha','max_sampled_tokens','prefix_tokens','continue_tokens','answer_tokens','checkpoint_every'):
            if type(getattr(self,key))is not int or getattr(self,key)<1:raise ValueError(f'invalid {key}')
        if self.k<2 and self.algorithm!='sft_answers':raise ValueError('at least two independent outcomes required')
        if self.k>16 or self.roots>64:raise ValueError('reference trainer resource limit')
        if self.prompt not in {'direct','cot','cod'}:raise ValueError('explicit shared prompt required')
        if not 0<=self.cost_weight<1 or self.quality_warmup<0 or self.cost_ramp<1:raise ValueError('invalid quality/cost curriculum')
        if not all(math.isfinite(x) and x>0 for x in (self.learning_rate,self.temperature)):raise ValueError('invalid optimizer/sampling parameters')
        for key in ('load_in_4bit','activation_offload','offload_kv_cache'):
            if type(getattr(self,key)) is not bool:raise ValueError(f'invalid {key}')
        if not math.isfinite(self.max_gpu_memory_gib) or self.max_gpu_memory_gib<=0:raise ValueError('invalid GPU memory limit')


def beta_at(step,cfg):
    return cfg.cost_weight*min(1.,max(0.,(step-cfg.quality_warmup+1)/cfg.cost_ramp))


def utility(reward,cost,scale,beta,multiplier=0.):
    """Failed answers have constant utility; shorter failures earn no length bonus.

    Every success is better than every failure at the outcome level for beta<1.
    This is NOT a guarantee that expected accuracy cannot decline during SGD.
    """
    if reward not in (0,1) or cost<0 or scale<=0 or not 0<=beta<1:raise ValueError('invalid utility arguments')
    return (1+multiplier)*reward-beta*(min(cost/scale,1.) if reward else 1.)


def advantages(rewards,algorithm):
    if len(rewards)<2 or any(not math.isfinite(r) for r in rewards):raise ValueError('invalid independent reward group')
    n=len(rewards);mean=statistics.mean(rewards)
    if algorithm=='rloo':return [(n*r-sum(rewards))/(n-1) for r in rewards]
    if algorithm=='drgrpo_reference':return [r-mean for r in rewards]
    if algorithm=='grpo_reference':
        std=statistics.pstdev(rewards)
        return [(r-mean)/(std+1e-8) for r in rewards]
    raise ValueError('algorithm has no group baseline')


def quality_credit(fork,craft_cfg,beta):
    """Reuse CRAFT's exact branch estimator under the new bounded cost objective."""
    from rl_craft.core import credit
    transformed=replace(fork,arms=tuple(tuple(replace(b,cost=min(b.cost,craft_cfg.cost_scale) if b.reward else craft_cfg.cost_scale) for b in arm) for arm in fork.arms))
    c=credit(transformed,replace(craft_cfg,cost_weight=beta))
    true_cost=sum(fork.probabilities[a]*statistics.mean(b.cost for b in fork.arms[a]) for a in (0,1))
    return replace(c,expected_cost=true_cost)


def degrpo_reference_loss(old_log_probs,log_probs,advantage,mask,alpha=.001,clip=.2):
    """Loss-level audit of Thinkless's split control/response weighting.

    Independent implementation of the inspected algorithm; NOT its full training
    recipe, warmup, special tokens or data. Input tensors have shape [B,T].
    """
    import torch
    if old_log_probs.shape!=log_probs.shape or advantage.shape!=log_probs.shape or mask.shape!=log_probs.shape or log_probs.ndim!=2 or log_probs.shape[1]<2:
        raise ValueError('aligned [batch,tokens>=2] tensors required')
    if alpha<0 or not 0<clip<1 or not torch.isfinite(log_probs).all() or not torch.isfinite(old_log_probs).all():raise ValueError('invalid policy loss inputs')
    ratio=(log_probs-old_log_probs.detach()).exp()
    loss=torch.maximum(-advantage.detach()*ratio,-advantage.detach()*ratio.clamp(1-clip,1+clip))
    control=mask.clone();control[:,1:]=0
    response=mask.clone();response[:,0]=0
    if control.sum()<=0 or response.sum()<=0:raise ValueError('both control and response observations required')
    return alpha*(loss*control).sum()/control.sum()+(loss*response).sum()/response.sum()


class ModelBridge:
    """Use the repo's GPU-tested sampling/logprob code and Mapping tokenizer fix."""
    def __init__(self,path,cfg,device='cuda:0',adapter=None):
        from rl_craft.core import Config
        from rl_craft.hf_backend import HFBackend
        self.cfg=cfg
        self.inner=HFBackend(path,Config(rank=cfg.rank,alpha=cfg.alpha,max_context=cfg.max_context,
                                        temperature=cfg.temperature),device,adapter,
                             load_in_4bit=cfg.load_in_4bit, activation_offload=cfg.activation_offload,
                             offload_kv_cache=cfg.offload_kv_cache)
        self.model=self.inner.model

    def context(self,question):
        from collections.abc import Mapping
        ids=self.inner.tokenizer.apply_chat_template([{'role':'user','content':question+'\n\n'+SUFFIXES[self.cfg.prompt]}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if isinstance(ids,Mapping):ids=ids['input_ids']
        if len(ids)+self.cfg.max_tokens>self.cfg.max_context:raise ValueError('training context overflow; no skipped cases')
        return tuple(ids)

    def sample(self,question,seed):
        s=self.inner.sample(self.context(question),self.cfg.max_tokens,self.cfg.temperature,seed)
        return s,self.inner.decode(s.tokens,final=True)

    def log_prob(self,segment):return self.inner.log_prob(segment,self.cfg.temperature)

    def supervised(self,question,target):
        from rl_craft.core import Segment
        ids=self.inner.tokenizer.encode('Final answer: '+target,add_special_tokens=False)+[self.inner.end_ids[0]]
        if len(ids)>self.cfg.max_tokens:raise ValueError('gold output over supervised cap')
        return Segment(self.context(question),tuple(ids),True)

    def save(self,path):self.inner.save_adapter(path)


class ContentTrainer:
    """A shared LoRA/data/prompt/runtime controlled comparison, not paper rebranding."""
    def __init__(self,backend,cases,cfg,scorer):
        import torch
        if not cases or any(c.split!='train' or c.source_split!='train' for c in cases):
            raise ValueError('new training controls use official training sources only')
        self.backend,self.cases,self.cfg,self.scorer=backend,cases,cfg,scorer
        self.parameters=[p for n,p in backend.model.named_parameters() if p.requires_grad]
        if not self.parameters or any('lora_' not in n for n,p in backend.model.named_parameters() if p.requires_grad):
            raise ValueError('only LoRA parameters may be optimized')
        backend.model.eval()
        self.optimizer=torch.optim.AdamW(self.parameters,lr=cfg.learning_rate,weight_decay=0.)
        self.rng=random.Random(cfg.seed);self.step_number=0;self.generated=0
        self.by_task={t:[c for c in cases if c.dataset+'/'+c.task==t] for t in sorted({c.dataset+'/'+c.task for c in cases})}

    def step(self):
        import torch
        cfg=self.cfg;start=time.perf_counter();k=1 if cfg.algorithm=='sft_answers' else cfg.k
        n=min(cfg.roots,(cfg.max_sampled_tokens-self.generated)//(k*cfg.max_tokens))
        if n<1:raise StopIteration('insufficient reserved action-token budget')
        selected=[self.rng.choice(self.by_task[self.rng.choice(list(self.by_task))]) for _ in range(n)]
        batch=[];correct=[];prefill=0;count=0;beta=beta_at(self.step_number,cfg)
        # ALL sampling before any weight update. No dropout, no top-p truncation.
        for j,c in enumerate(selected):
            group=[]
            for sample in range(k):
                if cfg.algorithm=='sft_answers':segment=self.backend.supervised(c.question,c.target);prediction=c.target;reward=1.
                else:
                    segment,prediction=self.backend.sample(c.question,int(digest([cfg.seed,self.step_number,j,sample,c.id])[:8],16)&0x7fffffff)
                    reward=float(bool(self.scorer(prediction,c)) and segment.terminated)
                group.append((segment,utility(reward,len(segment.tokens),cfg.max_tokens,beta),reward))
                count+=len(segment.tokens);prefill+=len(segment.context);correct.append(reward)
            batch.append(group)
        self.optimizer.zero_grad(set_to_none=True);loss_value=0.
        for group in batch:
            adv=[1.] if cfg.algorithm=='sft_answers' else advantages([u for _,u,_ in group],cfg.algorithm)
            for (segment,_,_),a in zip(group,adv):
                if a == 0:
                    continue  # Exactly zero gradient; generated tokens still count.
                # GRPO has per-response token normalization; Dr.GRPO a fixed cap.
                denom=len(segment.tokens) if cfg.algorithm in {'grpo_reference','sft_answers'} else cfg.max_tokens if cfg.algorithm=='drgrpo_reference' else 1
                loss=-self.backend.log_prob(segment)*a/(n*k*denom)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss_value+=float(loss.detach());loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True));self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.parameters):raise FloatingPointError('nonfinite adapter')
        self.generated+=count;self.step_number+=1
        return {'step':self.step_number,'algorithm':cfg.algorithm,'beta':beta,'roots':n,'sampled_tokens':count if cfg.algorithm!='sft_answers' else 0,
                'supervised_target_tokens':count if cfg.algorithm=='sft_answers' else 0,
                'cumulative_action_tokens':self.generated,'sampled_case_ids':[c.id for c in selected],
                'prefill_tokens':prefill,'train_accuracy':statistics.mean(correct) if cfg.algorithm!='sft_answers' else None,
                'loss':loss_value,'gradient_norm':norm,'seconds':time.perf_counter()-start,
                'accuracy_note':'training outcomes, not an evaluation result'}


class CraftQualityTrainer:
    """Same-prefix paired CRAFT estimator + accuracy-first warmup and bounded cost.

    Unlike the old benchmark-prefix trainer, this explicitly accepts official
    external training splits; it does not relabel their indices as 0..24.
    """
    def __init__(self,backend,cases,cfg,scorer):
        import torch
        from rl_craft.core import Config,Example,Scheduler
        from rl_craft.trainer import lora_parameters
        if not cases or any(c.split!='train' or c.source_split!='train' for c in cases):raise ValueError('official training only')
        self.cfg,self.scorer=cfg,scorer;self.lookup={c.id:c for c in cases}
        self.core=Config(roots_per_step=cfg.roots,samples_per_arm=cfg.k,prefix_tokens=cfg.prefix_tokens,
                        continue_tokens=cfg.continue_tokens,answer_tokens=cfg.answer_tokens,max_context=cfg.max_context,
                        temperature=cfg.temperature,cost_scale=cfg.prefix_tokens+cfg.continue_tokens+cfg.answer_tokens,
                        cost_weight=cfg.cost_weight,rank=cfg.rank,alpha=cfg.alpha,quality_dual=False,seed=cfg.seed)
        self.backend=backend.inner;self.backend.config=self.core;self.backend.model.eval()
        examples=[Example(c.id,c.dataset+'/'+c.task,c.question,c.target,i) for i,c in enumerate(cases)]
        self.examples={e.key:e for e in examples};self.scheduler=Scheduler(examples,self.core)
        self.parameters=lora_parameters(self.backend.model)
        self.optimizer=torch.optim.AdamW(self.parameters,lr=cfg.learning_rate,weight_decay=0.)
        self.rng=random.Random(cfg.seed);self.step_number=0;self.generated=0

    def step(self):
        import torch
        from rl_craft.trainer import collect,apply_credit
        cfg=self.cfg;start=time.perf_counter()
        n=min(cfg.roots,(cfg.max_sampled_tokens-self.generated)//self.core.worst_tree_tokens)
        if not n:raise StopIteration('insufficient reserved tree budget')
        draws=self.scheduler.draw(self.rng,n);forks=[]
        for j,(key,importance) in enumerate(draws):
            score=lambda pred,gold,c=self.lookup[key]:self.scorer(pred,c)
            forks.append(collect(self.backend,self.examples[key],self.core,self.scheduler,importance,self.step_number,j,score))
        beta=beta_at(self.step_number,cfg);credits=[quality_credit(f,self.core,beta) for f in forks]
        self.optimizer.zero_grad(set_to_none=True)
        loss=sum(apply_credit(self.backend,f,c,n) for f,c in zip(forks,credits))
        norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True));self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.parameters):raise FloatingPointError('nonfinite adapter')
        self.scheduler.update(forks,credits)
        generated=sum(len(f.prefix.tokens)+sum(len(s.tokens) for arm in f.arms for b in arm for s in b.segments) for f in forks)
        self.generated+=generated;self.step_number+=1
        return {'step':self.step_number,'algorithm':'craft_quality','beta':beta,'roots':n,'sampled_tokens':generated,
                'cumulative_action_tokens':self.generated,'sampled_case_ids':[f.key for f in forks],'train_accuracy_ht':sum(f.importance*c.accuracy for f,c in zip(forks,credits))/n,
                'prefill_tokens':sum(len(f.prefix.context)+len(f.gate_context)+sum(len(s.context) for arm in f.arms for b in arm for s in b.segments) for f in forks),
                'loss':loss,'gradient_norm':norm,'seconds':time.perf_counter()-start,
                'accuracy_note':'HT training estimate, not evaluation; quality dual disabled to isolate curriculum'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--model-path',required=True);p.add_argument('--initial-adapter');p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--bbeh-scorer',type=Path)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    try:
        import torch
        from rl_craft.data import model_identity
        from comparison_runtime import (limit_gpu_memory, latest_checkpoint, save_training_checkpoint,
                                        restore_training_state, adapter_alias)
        cfg=TrainConfig(**json.loads(args.config.read_text()));cases=load_cases(args.data)
        if any(c.split!='train' or c.source_split!='train' for c in cases):raise ValueError('use an explicit train-only file, not mixed dev/test data')
        scorer=Scorer(cases,args.bbeh_scorer)
        initial_files={str(f.relative_to(args.initial_adapter)):file_hash(f) for f in Path(args.initial_adapter).rglob('*') if f.is_file()} if args.initial_adapter else None
        protocol={'config':asdict(cfg),'data_digest':digest([asdict(c) for c in cases]),'scorer':scorer.identity,
                  'model_identity':model_identity(Path(args.model_path)),'initial_adapter_files':initial_files,
                  'environment':environment(),'training_code_sha256':file_hash(__file__),
                  'comparison_code_sha256':source_hashes(),
                  'implementation':'CRAFT-Q proposal' if cfg.algorithm=='craft_quality' else 'matched-runtime reference, not full upstream reproduction'}
        protocol.update(schema=2, precision='NF4-double-quant/BF16-compute' if cfg.load_in_4bit else 'BF16',
                        data_file_sha256=file_hash(args.data), model_path=str(Path(args.model_path).resolve()))
        checkpoint=None
        if args.resume:
            if json.loads((args.output/'protocol.json').read_text()) != protocol:
                raise ValueError('resume protocol differs: model/data/code/environment/config changed')
            checkpoint=latest_checkpoint(args.output,protocol)
            if (args.output/'summary.json').exists():
                report=json.loads((args.output/'summary.json').read_text())
                if report.get('status')!='complete' or report.get('protocol_digest')!=digest(protocol):
                    raise ValueError('invalid completed training summary')
                print(json.dumps(report,indent=2));return 0
        else:
            args.output.mkdir(parents=True,exist_ok=False)
            write_json(args.output/'protocol.json',protocol)
        limit_gpu_memory(args.device,cfg.max_gpu_memory_gib)
        torch.manual_seed(cfg.seed)
        backend=ModelBridge(args.model_path,cfg,args.device,str(checkpoint/'adapter') if checkpoint else args.initial_adapter)
        engine=(CraftQualityTrainer if cfg.algorithm=='craft_quality' else ContentTrainer)(backend,cases,cfg,scorer)
        metrics=[];reason='steps-complete'
        if checkpoint:
            restore_training_state(engine,torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True))
            metrics=json.loads((checkpoint/'metrics.json').read_text())
            if (args.output/'metrics.jsonl').exists():
                os.replace(args.output/'metrics.jsonl',args.output/f'metrics-before-resume-{time.time_ns()}.jsonl')
        else:
            checkpoint=save_training_checkpoint(engine,backend,args.output,protocol,metrics)
        started=time.perf_counter()
        try:
            with (args.output/'metrics.jsonl').open('x') as f:
                for row in metrics:f.write(json.dumps(row)+'\n')
                f.flush()
                while engine.step_number<cfg.steps:
                    if torch.device(args.device).type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
                    try:row=engine.step()
                    except StopIteration as exc:reason=str(exc);break
                    if torch.device(args.device).type=='cuda':
                        row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
                        row['peak_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
                    f.write(json.dumps(row)+'\n');f.flush();metrics.append(row)
                    print(json.dumps(row,sort_keys=True),flush=True)
                    if engine.step_number%cfg.checkpoint_every==0:
                        checkpoint=save_training_checkpoint(engine,backend,args.output,protocol,metrics)
                        adapter_alias(args.output,f'adapter-{engine.step_number:06d}',checkpoint)
            if int(checkpoint.name.split('-')[-1])!=engine.step_number:
                checkpoint=save_training_checkpoint(engine,backend,args.output,protocol,metrics)
            adapter_alias(args.output,'adapter-final',checkpoint)
        except BaseException as exc:
            write_json(args.output/f'failure-{time.time_ns()}.json',
                       {'type':type(exc).__name__,'message':str(exc),'completed_steps':engine.step_number,
                        'completed_action_tokens':engine.generated,'invocation_seconds':time.perf_counter()-started,
                        'cost_accounting':'failed in-flight work may be incomplete; preserve this when comparing total compute'})
            raise
        write_json(args.output/'summary.json',{'status':'complete','steps':engine.step_number,'action_tokens':engine.generated,'stop_reason':reason,
                    'protocol_digest':digest(protocol),'no_evaluation_claim':True,'checkpoint':checkpoint.name,
                    'had_failed_invocations':bool(list(args.output.glob('failure-*.json'))),
                    'this_invocation_seconds':time.perf_counter()-started})
        print(json.dumps(metrics[-1] if metrics else {'status':reason},indent=2));return 0
    except (ValueError,OSError,ImportError) as exc:p.exit(2,f'error: {exc}\n')


if __name__=='__main__':raise SystemExit(main())
