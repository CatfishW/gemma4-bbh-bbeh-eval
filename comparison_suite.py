"""Paper-grounded, offline-first LLM augmentation comparisons.

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


from comparison_data import (Case, SCORERS, Scorer, assert_disjoint, convert, digest,
    file_hash, final_text, load_cases, numeric, read_rows, split_training,
    validate_cases, vote_key, write_json)
from comparison_methods import APIBackend, Generation, LocalBackend, Method, SUFFIXES, run_method


def source_hashes():
    root=Path(__file__).parent
    paths=list(root.glob('comparison_*.py'))+list((root/'rl_craft').glob('*.py'))
    paths.extend([root/'rl/modeling.py',root/'eval_benchmarks.py'])
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(paths)}


def environment():
    versions={}
    for name in ('torch','transformers','peft','bitsandbytes','accelerate','safetensors','pyarrow','math-verify','latex2sympy2_extended','vllm','sglang'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'python':sys.version.split()[0],'packages':versions}


def validate_identity(identity):
    required={'base_model_sha256','checkpoint_sha256','hardware_id','precision','serving_stack','load_profile','study_id'}
    if not isinstance(identity,dict) or not required<=identity.keys() or any(not isinstance(identity[k],str) or not identity[k].strip() for k in required):
        raise ValueError('incomplete model/environment identity')
    if any(not re.fullmatch('[0-9a-f]{64}',identity[k]) for k in ('base_model_sha256','checkpoint_sha256')):
        raise ValueError('model identities must be SHA-256 digests')
    return identity


def run(cases,method,backend,output,identity,scorer,seed=7,resume=False):
    validate_identity(identity)
    output=Path(output)
    cfg={'schema':1,'method':asdict(method),'implementation':'same-model-transfer-not-original-paper-reproduction',
         'data_digest':digest([asdict(c) for c in cases]),'scorer':scorer.identity,'identity':identity,
         'environment':environment(),'seed':seed,'case_ids':[c.id for c in cases],
         'source_sha256':file_hash(__file__),'code_sha256':source_hashes(),'status':'started','concurrency':1}
    from comparison_runtime import resume_predictions
    results=resume_predictions(output,cfg,cases,resume)
    if (output/'summary.json').exists():return json.loads((output/'summary.json').read_text())
    start=time.perf_counter()
    with (output/'predictions.jsonl').open('a' if resume else 'x',encoding='utf-8') as f:
        for c in cases[len(results):]:
            before=time.perf_counter();result=run_method(c.question,method,backend,seed)
            inference_seconds=time.perf_counter()-before
            score_start=time.perf_counter()
            correct=bool(scorer(result['prediction'],c)) if result['error'] is None else False
            scoring_seconds=time.perf_counter()-score_start
            result.update(id=c.id,dataset=c.dataset,task=c.task,group=c.group,case_digest=digest(asdict(c)),
                    correct=correct, scoring_seconds=scoring_seconds,
                    elapsed_seconds=inference_seconds)
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();results.append(result)
            if len(results)==1 or len(results)%25==0:
                print(json.dumps({'completed_cases':len(results),'total_cases':len(cases)}),flush=True)
    report=summarize(results);report.update(this_invocation_seconds=time.perf_counter()-start,resumed=resume,protocol_digest=digest(cfg),status='complete',predictions_sha256=file_hash(output/'predictions.jsonl'))
    write_json(output/'summary.json',report)
    return report


def summarize(rows):
    if not rows: raise ValueError('cannot summarize empty run')
    by_task=defaultdict(list)
    for r in rows: by_task[f"{r['dataset']}/{r['task']}"].append(r)
    acc=lambda rs:sum(r['correct'] for r in rs)/len(rs)
    known=lambda key:all(type(r.get(key)) is int and r[key]>=0 for r in rows)
    times=sorted(r['elapsed_seconds'] for r in rows)
    return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'accuracy':acc(rows),
            'macro_task_accuracy':statistics.mean(acc(rs) for rs in by_task.values()),
            'per_task':{t:{'n':len(rs),'accuracy':acc(rs)} for t,rs in by_task.items()},
            'mean_seconds':statistics.mean(times),'p95_seconds':times[math.ceil(.95*len(times))-1],
            'errors':sum(r.get('error') is not None for r in rows),
            'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rows) if known('output_tokens') else None,
            'mean_input_tokens':statistics.mean(r['input_tokens'] for r in rows) if known('input_tokens') else None,
            'truncated_calls':sum(c['generation'].get('finish_reason')=='length' for r in rows for c in r.get('calls',[]))}


def completed(path):
    path=Path(path);cfg=json.loads((path/'protocol.json').read_text());summary=json.loads((path/'summary.json').read_text())
    rows=read_rows(path/'predictions.jsonl')
    if summary.get('status')!='complete' or summary.get('protocol_digest')!=digest(cfg) or summary.get('predictions_sha256')!=file_hash(path/'predictions.jsonl'):
        raise ValueError('incomplete run or protocol digest mismatch')
    if len(rows)!=len(cfg['case_ids']) or {r['id'] for r in rows}!=set(cfg['case_ids']) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('missing/extra/duplicate predictions')
    if summary['n']!=len(rows) or summary['correct']!=sum(r['correct'] for r in rows):
        raise ValueError('summary mismatch')
    for r in rows:
        if type(r.get('correct')) is not bool or not math.isfinite(r['elapsed_seconds']) or r['elapsed_seconds']<0:
            raise ValueError('invalid result types/cost')
    return cfg,rows


def mcnemar(wins,losses):
    n=wins+losses
    if not n:return 1.
    k=min(wins,losses)
    logs=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
    high=max(logs)
    return min(1.,2*math.exp(high)*sum(math.exp(v-high) for v in logs))


def holm(values):
    order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);prior=0.
    for rank,i in enumerate(order): prior=max(prior,min(1.,(len(values)-rank)*values[i]));out[i]=prior
    return out


def paired(a,b,reps=2000,seed=7,cluster='group'):
    if reps<100:raise ValueError('at least 100 bootstrap replicates required')
    left={r['id']:r for r in a};right={r['id']:r for r in b}
    if len(left)!=len(a) or len(right)!=len(b) or set(left)!=set(right):raise ValueError('paired coverage mismatch')
    groups=defaultdict(list)
    for k,r in left.items():
        s=right[k]
        if any(r[x]!=s[x] for x in ('case_digest','group','dataset','task')):raise ValueError('paired cases differ')
        groups[r['group'] if cluster=='group' else f"{r['dataset']}/{r['task']}"].append(k)
    keys=sorted(left);gkeys=sorted(groups);rng=random.Random(seed)
    delta=sum(int(right[k]['correct'])-int(left[k]['correct']) for k in keys)/len(keys)
    wins=sum(not left[k]['correct'] and right[k]['correct'] for k in keys)
    losses=sum(left[k]['correct'] and not right[k]['correct'] for k in keys)
    boot=[];ratios=[]
    for _ in range(reps):
        sample=[k for g in rng.choices(gkeys,k=len(gkeys)) for k in groups[g]]
        boot.append(sum(int(right[k]['correct'])-int(left[k]['correct']) for k in sample)/len(sample))
        denominator=sum(left[k]['elapsed_seconds'] for k in sample)
        if denominator>0:ratios.append(sum(right[k]['elapsed_seconds'] for k in sample)/denominator)
    def interval(xs):
        xs=sorted(xs)
        return [xs[int(.025*(len(xs)-1))],xs[int(.975*(len(xs)-1))]] if xs else None
    return {'n':len(keys),'groups':len(groups),'accuracy_delta':delta,'paired_wins':wins,'paired_losses':losses,
            'accuracy_ci95':interval(boot),'latency_ratio_ci95':interval(ratios),'mcnemar_p':mcnemar(wins,losses),
            'bootstrap':f'paired {cluster}-cluster percentile; inference-seed/training-seed uncertainty not included',
            'replicates':reps}


def compare(paths,baseline,reps=2000):
    bcfg,brows=completed(baseline);reports=[]
    for path in paths:
        cfg,rows=completed(path)
        for key in ('data_digest','scorer','seed'):
            if cfg.get(key)!=bcfg.get(key):raise ValueError(f'incomparable {key}')
        report=paired(brows,rows,reps)
        report['task_cluster_accuracy_ci95']=paired(brows,rows,reps,cluster='task')['accuracy_ci95']
        cost_keys=('hardware_id','precision','serving_stack','load_profile')
        report['controlled_latency']=all(cfg['identity'].get(k) and cfg['identity'].get(k)==bcfg['identity'].get(k) for k in cost_keys)
        report.update(candidate=str(path),baseline=str(baseline),same_base=cfg['identity'].get('base_model_sha256')==bcfg['identity'].get('base_model_sha256'))
        report['comparison_class']='matched-base' if report['same_base'] else 'different-model-descriptive'
        report['sota_claim']=False
        reports.append(report)
    for r,p in zip(reports,holm([r['mcnemar_p'] for r in reports])):r['holm_p']=p
    return reports


def import_run(cases,source,output,metadata,scorer):
    """Import author-run module outputs; recompute correctness, never trust supplied scores."""
    required={'method','upstream_commit','implementation','identity','seed','all_calls_accounted'}
    if not required<=metadata.keys() or not re.fullmatch('[0-9a-f]{40}',metadata['upstream_commit']):
        raise ValueError('upstream run needs a full commit and execution/identity metadata')
    validate_identity(metadata['identity'])
    if type(metadata['all_calls_accounted']) is not bool:raise ValueError('all_calls_accounted must be boolean')
    raw=read_rows(source);lookup={r['id']:r for r in raw}
    if len(lookup)!=len(raw) or set(lookup)!={c.id for c in cases}:raise ValueError('external cohort mismatch')
    rows=[]
    for c in cases:
        r=lookup[c.id]
        if r.get('case_digest')!=digest(asdict(c)):raise ValueError('external input identity not established')
        if not isinstance(r.get('prediction'),str):raise ValueError('missing external prediction')
        seconds=r.get('elapsed_seconds')
        if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:raise ValueError('external latency required')
        if any(r.get(k) is not None and (type(r[k]) is not int or r[k]<0) for k in ('output_tokens','input_tokens')):raise ValueError('invalid external token usage')
        usage={key:r.get(key) if metadata['all_calls_accounted'] is True else None for key in ('output_tokens','input_tokens')}
        rows.append({'id':c.id,'dataset':c.dataset,'task':c.task,'group':c.group,'case_digest':digest(asdict(c)),
            'prediction':r['prediction'],'correct':False if r.get('error') else bool(scorer(r['prediction'],c)),
            'elapsed_seconds':seconds,'error':r.get('error'),'calls':r.get('calls',[]),**usage})
    path=Path(output);path.mkdir(parents=True,exist_ok=False)
    cfg={'schema':1,'data_digest':digest([asdict(c) for c in cases]),'scorer':scorer.identity,'case_ids':[c.id for c in cases],
         'seed':metadata['seed'],'identity':metadata['identity'],'external':metadata,'source_sha256':file_hash(source)}
    write_json(path/'protocol.json',cfg)
    with (path/'predictions.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result=summarize(rows);result.update(status='complete',protocol_digest=digest(cfg),predictions_sha256=file_hash(path/'predictions.jsonl'))
    write_json(path/'summary.json',result);return result


def write_cases(path,cases,provenance):
    cases=validate_cases(cases)
    path=Path(path);sidecar=Path(str(path)+'.manifest.json')
    if path.exists() or sidecar.exists():raise FileExistsError('dataset or manifest exists')
    with path.open('x',encoding='utf-8') as f:
        for c in cases:f.write(json.dumps(asdict(c),ensure_ascii=False)+'\n')
    manifest={'n':len(cases),'data_digest':digest([asdict(c) for c in cases]),
              'file_sha256':file_hash(path),'provenance':provenance}
    write_json(sidecar,manifest);return manifest


def export_legacy(root,benchmarks,split):
    import eval_benchmarks as legacy
    loaders={'bbh':legacy.load_bbh,'bbeh':legacy.load_bbeh,'usr':legacy.load_unpuzzles_simple_reasoning}
    if not benchmarks or len(set(benchmarks))!=len(benchmarks) or any(b not in loaders for b in benchmarks):
        raise ValueError('invalid benchmark names')
    # Preserve puzzle families using source puzzle names, never task-local indices.
    groups={}
    if 'usr' in benchmarks:
        for fname in ('unpuzzles.json','shifted_unpuzzles.json'):
            file=Path(root)/'unpuzzles_and_simple_reasoning/datasets'/fname
            for r in json.loads(file.read_text()):
                name=str(r.get('puzzle_name','')).strip()
                group='usr-puzzle:'+digest(name or str(r.get('original_puzzle','')))
                for key in ('original_puzzle','unpuzzle','shifted_unpuzzle'):
                    if r.get(key):
                        q=str(r[key]).strip()
                        if name:q=f'Puzzle: {name}\n\n{q}'
                        if q in groups and groups[q]!=group:raise ValueError('ambiguous source puzzle grouping')
                        groups[q]=group
    output=[]
    for b in benchmarks:
        rows=loaders[b](Path(root),None)
        revision=digest([asdict(r) for r in rows])
        for e in rows:
            keep=(0<=e.index<25) if split=='calibration' else (25<=e.index<50) if split=='validation' else e.index>=50
            if not keep:continue
            group=groups.get(e.input,digest(e.input.strip()))
            if b=='usr' and ('unpuzzles/' in e.task) and e.input not in groups:
                raise ValueError('missing source puzzle group; refusing unsafe split')
            output.append(Case(f'{b}/{e.task}/{e.index}',b,e.task,f'{b}/{group}',e.input,e.target,
                'test' if split=='test' else 'dev','evaluation-only',revision,'bbeh_official' if b=='bbeh' else 'legacy'))
    return validate_cases(output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare');prep.add_argument('--input',required=True,type=Path);prep.add_argument('--output',required=True,type=Path)
    prep.add_argument('--format',choices=['canonical','gsm8k','math','math500','aime','gpqa','mmlu_pro','bbh','bbeh','svamp','deepscaler'],required=True)
    prep.add_argument('--dataset',required=True);prep.add_argument('--split',choices=['train','dev','test'],required=True)
    prep.add_argument('--source-split',choices=['train','validation','test','evaluation-only'],required=True)
    prep.add_argument('--revision',required=True);prep.add_argument('--scorer',choices=sorted(SCORERS));prep.add_argument('--group-field')
    prep.add_argument('--expected-count',required=True,type=int);prep.add_argument('--dev-fraction',type=float)
    for cmd in ('run','import-run'):
        q=subs.add_parser(cmd);q.add_argument('--data',required=True,type=Path);q.add_argument('--output',required=True,type=Path)
        q.add_argument('--identity',required=True,type=Path);q.add_argument('--allow-test',action='store_true');q.add_argument('--bbeh-scorer',type=Path)
        if cmd=='run':
            q.add_argument('--method',required=True,type=Path);q.add_argument('--seed',type=int,default=7)
            source=q.add_mutually_exclusive_group(required=True);source.add_argument('--base-url');source.add_argument('--model-path')
            q.add_argument('--model',default='SubTokenLLM-E2B');q.add_argument('--adapter');q.add_argument('--device',default='cuda:0')
            q.add_argument('--resume',action='store_true');q.add_argument('--load-in-4bit',action='store_true')
            q.add_argument('--offload-kv-cache',action='store_true');q.add_argument('--max-gpu-memory-gib',type=float,default=8.)
            q.add_argument('--loader',choices=['legacy_gemma','causal'],default='legacy_gemma');q.add_argument('--max-context',type=int,default=16384)
        else:q.add_argument('--predictions',required=True,type=Path)
    q=subs.add_parser('compare');q.add_argument('--baseline',required=True,type=Path);q.add_argument('--candidate',required=True,type=Path,action='append');q.add_argument('--output',required=True,type=Path);q.add_argument('--replicates',type=int,default=2000)
    q=subs.add_parser('doctor');q.add_argument('--output',required=True,type=Path)
    q=subs.add_parser('select');q.add_argument('--data',required=True,type=Path);q.add_argument('--split',choices=['train','dev','test'],required=True);q.add_argument('--output',required=True,type=Path)
    q=subs.add_parser('merge');q.add_argument('--data',required=True,type=Path,action='append');q.add_argument('--expected-count',required=True,type=int);q.add_argument('--output',required=True,type=Path)
    q=subs.add_parser('export-legacy');q.add_argument('--datasets-root',required=True,type=Path);q.add_argument('--benchmarks',default='bbh,bbeh,usr');q.add_argument('--legacy-split',choices=['calibration','validation','test'],required=True);q.add_argument('--expected-count',required=True,type=int);q.add_argument('--output',required=True,type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=='doctor': result=environment();write_json(args.output,result)
        elif args.command=='select':
            result=write_cases(args.output,[c for c in load_cases(args.data) if c.split==args.split],{'source':file_hash(args.data),'selected_split':args.split})
        elif args.command=='merge':
            cases=[c for pth in args.data for c in load_cases(pth)]
            if len(cases)!=args.expected_count:raise ValueError('merge count mismatch')
            result=write_cases(args.output,cases,{'inputs':[file_hash(pth) for pth in args.data]})
        elif args.command=='export-legacy':
            cases=export_legacy(args.datasets_root,args.benchmarks.split(','),args.legacy_split)
            if len(cases)!=args.expected_count:raise ValueError('legacy cohort count mismatch')
            result=write_cases(args.output,cases,{'legacy_split':args.legacy_split,'root':str(args.datasets_root),'historically_exposed':True})
        elif args.command=='prepare':
            cases=convert(args.input,args.format,args.dataset,args.split,args.source_split,args.revision,args.scorer,args.group_field)
            if len(cases)!=args.expected_count:raise ValueError('expected dataset count mismatch; refusing partial cohort')
            if args.dev_fraction is not None:cases=split_training(cases,args.dev_fraction)
            result=write_cases(args.output,cases,{'source_sha256':file_hash(args.input),'revision':args.revision,'source_split':args.source_split})
        elif args.command=='compare':result=compare(args.candidate,args.baseline,args.replicates);write_json(args.output,result)
        else:
            cases=load_cases(args.data)
            if any(c.split=='test' for c in cases) and not args.allow_test:raise ValueError('test requires --allow-test after protocol freeze')
            scorer=Scorer(cases,args.bbeh_scorer);identity=json.loads(args.identity.read_text())
            if args.command=='import-run':result=import_run(cases,args.predictions,args.output,identity,scorer)
            else:
                validate_identity(identity)
                method=Method(**json.loads(args.method.read_text()))
                if not args.base_url:
                    from rl_craft.data import model_identity
                    identity['base_model_sha256']=model_identity(Path(args.model_path))['digest']
                    adapter_hashes={str(f.relative_to(args.adapter)):file_hash(f) for f in sorted(Path(args.adapter).rglob('*')) if f.is_file()} if args.adapter else None
                    if args.adapter and not adapter_hashes:raise ValueError('empty adapter directory')
                    identity['checkpoint_sha256']=digest(adapter_hashes) if adapter_hashes is not None else identity['base_model_sha256']
                if not args.base_url:
                    identity['precision']='NF4-double-quant/BF16-compute' if args.load_in_4bit else 'BF16'
                    identity['runtime_options']={'loader':args.loader,'max_context':args.max_context,'offload_kv_cache':args.offload_kv_cache,'max_gpu_memory_gib':args.max_gpu_memory_gib}
                backend=APIBackend(args.base_url,args.model) if args.base_url else LocalBackend(args.model_path,args.device,args.adapter,args.max_context,args.loader,load_in_4bit=args.load_in_4bit,offload_kv_cache=args.offload_kv_cache,max_gpu_memory_gib=args.max_gpu_memory_gib)
                result=run(cases,method,backend,args.output,identity,scorer,args.seed,args.resume)
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    except (ValueError,OSError,ImportError) as exc:p.exit(2,f'error: {exc}\n')


if __name__=='__main__':raise SystemExit(main())
