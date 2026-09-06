"""Evaluate existing CRAFT / new CRAFT-Q adapters with the common comparison scorer.

No training, gold-based stopping, output repair, or reward feedback at evaluation.
Raw original pilot artifacts and their scoring are never rewritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from comparison_suite import (Scorer, digest, environment, file_hash, load_cases,
                               summarize, write_json)


def run(cases, backend, config, output, identity, scorer, seed=7, gate='sample', resume=False):
    from rl_craft.trainer import predict
    path=Path(output)
    protocol={'schema':1,'method':{'name':'craft','gate':gate,'config':asdict(config)},
              'data_digest':digest([asdict(c) for c in cases]),'scorer':scorer.identity,
              'seed':seed,'case_ids':[c.id for c in cases],'identity':identity,
              'environment':environment(),'bridge_sha256':file_hash(__file__),
              'implementation':'existing CRAFT staged policy with common post-hoc scoring',
              'source_sha256':file_hash(Path(__file__).parent/'rl_craft/trainer.py'),
              'code_sha256':__import__('comparison_suite').source_hashes()}
    from comparison_runtime import resume_predictions
    rows=resume_predictions(path,protocol,cases,resume)
    if (path/'summary.json').exists():return json.loads((path/'summary.json').read_text())
    start=time.perf_counter()
    with (path/'predictions.jsonl').open('a' if resume else 'x',encoding='utf-8') as stream:
        for c in cases[len(rows):]:
            before=time.perf_counter();error=None;raw={}
            try:raw=predict(backend,c.question,config,seed=seed,mode=gate)
            except (ValueError,RuntimeError) as exc:
                from comparison_methods import resource_failure
                if resource_failure(exc):raise
                error=f'{type(exc).__name__}: {exc}'
            elapsed=time.perf_counter()-before
            # All methods use the same scorer. Preserve both scorer correctness
            # and termination: official EM does not always demand an EOS token.
            score_start=time.perf_counter();answer=raw.get('prediction','')
            correct=bool(scorer(answer,c)) if error is None else False
            calls=[{'action':'stage','generation':{'finish_reason':'stop' if s['terminated'] else 'length'}}
                   for s in raw.get('segments',[])]
            row={'id':c.id,'dataset':c.dataset,'task':c.task,'group':c.group,
                 'case_digest':digest(asdict(c)),'prediction':answer,'correct':correct,
                 'elapsed_seconds':elapsed,'scoring_seconds':time.perf_counter()-score_start,
                 'input_tokens':raw.get('prefill_tokens'),'output_tokens':raw.get('generated_tokens'),
                 'error':error,'calls':calls,'raw':raw,
                 'model_forwards_note':'reported prefill includes the gate; forward-kernel count is not model-call count'}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();rows.append(row)
            if len(rows)==1 or len(rows)%25==0:
                print(json.dumps({'completed_cases':len(rows),'total_cases':len(cases)}),flush=True)
    report=summarize(rows);report.update(status='complete',protocol_digest=digest(protocol),
                        predictions_sha256=file_hash(path/'predictions.jsonl'),this_invocation_seconds=time.perf_counter()-start,resumed=resume)
    write_json(path/'summary.json',report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',type=Path,help='original atomic rl_craft checkpoint directory')
    source.add_argument('--training-run',type=Path,help='new comparison_training.py run with adapter-final')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--device',default='cuda:0')
    p.add_argument('--gate',choices=['sample','greedy','always-stop','always-continue'],default='sample')
    p.add_argument('--resume',action='store_true');p.add_argument('--load-in-4bit',action='store_true')
    p.add_argument('--max-context',type=int)
    p.add_argument('--offload-kv-cache',action='store_true');p.add_argument('--max-gpu-memory-gib',type=float,default=8.)
    p.add_argument('--allow-test',action='store_true');p.add_argument('--bbeh-scorer',type=Path)
    args=p.parse_args(argv)
    try:
        from rl_craft.cli import verify_checkpoint
        from rl_craft.core import Config
        from rl_craft.data import model_identity
        from rl_craft.hf_backend import HFBackend
        cases=load_cases(args.data)
        if any(c.split=='test' for c in cases) and not args.allow_test:raise ValueError('test requires --allow-test')
        actual=model_identity(args.model_path)
        if args.checkpoint:
            meta=verify_checkpoint(args.checkpoint)
            if meta['model_identity']!=actual:raise ValueError('base checkpoint mismatch')
            cfg=Config(**meta['config']);adapter=args.checkpoint/'adapter'
        else:
            meta=json.loads((args.training_run/'protocol.json').read_text())
            report=json.loads((args.training_run/'summary.json').read_text())
            if report.get('status')!='complete' or report.get('protocol_digest')!=digest(meta):raise ValueError('incomplete training run')
            if meta['model_identity']!=actual or meta['config']['algorithm']!='craft_quality':raise ValueError('not a matching CRAFT-Q run')
            t=meta['config'];cfg=Config(rank=t['rank'],alpha=t['alpha'],max_context=t['max_context'],
                     prefix_tokens=t['prefix_tokens'],continue_tokens=t['continue_tokens'],answer_tokens=t['answer_tokens'],
                     temperature=t['temperature'],quality_dual=False)
            from comparison_runtime import latest_checkpoint
            checkpoint=latest_checkpoint(args.training_run,meta)
            adapter=checkpoint/'adapter'
            if bool(t.get('load_in_4bit',False)) != args.load_in_4bit:raise ValueError('evaluation precision differs from training')
        if args.max_context:cfg=replace(cfg,max_context=args.max_context)
        identity=json.loads(args.identity.read_text())
        required={'hardware_id','precision','serving_stack','load_profile','study_id'}
        if not required<=identity.keys() or any(not isinstance(identity[k],str) or not identity[k] for k in required):raise ValueError('hardware/study identity required')
        adapter_files={str(f.relative_to(adapter)):file_hash(f) for f in sorted(adapter.rglob('*')) if f.is_file()}
        if not adapter_files:raise ValueError('empty adapter directory')
        identity.update(base_model_sha256=actual['digest'],checkpoint_sha256=digest(adapter_files))
        scorer=Scorer(cases,args.bbeh_scorer)
        from comparison_runtime import limit_gpu_memory
        limit_gpu_memory(args.device,args.max_gpu_memory_gib)
        identity['precision']='NF4-double-quant/BF16-compute' if args.load_in_4bit else 'BF16'
        identity['runtime_options']={'offload_kv_cache':args.offload_kv_cache,'max_gpu_memory_gib':args.max_gpu_memory_gib}
        backend=HFBackend(str(args.model_path),cfg,args.device,str(adapter),load_in_4bit=args.load_in_4bit,offload_kv_cache=args.offload_kv_cache)
        result=run(cases,backend,cfg,args.output,identity,scorer,args.seed,args.gate,args.resume)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,OSError,ImportError) as exc:p.exit(2,f'error: {exc}\n')


if __name__=='__main__':raise SystemExit(main())
