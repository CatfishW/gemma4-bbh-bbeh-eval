"""Durable sequential GPU-1 campaign; no service management or external writes.

The immutable plan lists every full-cohort run before execution. Failures stop the
queue with logs and verified resume artifacts; they are never scored as success.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from comparison_data import digest, file_hash, write_json, load_cases
from comparison_suite import source_hashes, write_cases
from comparison_training import TrainConfig

ALGORITHMS = ['craft_quality','rloo','grpo_reference','drgrpo_reference','sft_answers']
SEEDS = [7,17,27]


def atomic_json(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def build(root,model):
    root,model=Path(root).resolve(),Path(model).resolve();repo=Path(__file__).parent.resolve()
    campaign=root/'campaign-v1';campaign.mkdir(exist_ok=False)
    for name in ['configs','identities','logs','train','eval','comparisons']:(campaign/name).mkdir()
    data=root/'frozen-data-v1';math=data/'math-test.jsonl'
    if not math.exists():write_cases(math,load_cases(data/'gsm8k-test.jsonl')+load_cases(data/'math500-test.jsonl'),{'cohorts':'full GSM8K and MATH-500'})
    from rl_craft.data import model_identity
    model_id=model_identity(model)['digest']
    identity={'base_model_sha256':model_id,'checkpoint_sha256':model_id,
        'hardware_id':'benwulab-remote/physical-GPU-1/RTX6000-Ada-48GiB',
        'precision':'NF4-double-quant/BF16-compute','serving_stack':'transformers-5.5.4/torch-2.9.1/peft-0.20.0',
        'study_id':'sota-20260905-gpu1-nf4-v1'}
    jobs=[];evals={'math':{},'legacy':{}}
    scorer='/data/benwulab/gemma4-eval/datasets/bbeh/bbeh/evaluate.py'
    def add(name,argv,output,kind):
        jobs.append({'name':name,'argv':argv,'output':str(output),'kind':kind})
    def evaluation(label,cohort,training=None,method='cot',released=False):
        key=label+'-'+cohort;out=campaign/'eval'/key;inp=data/('math-test.jsonl' if cohort=='math' else 'legacy-test.jsonl')
        ident=campaign/'identities'/(key+'.json')
        write_json(ident,{**identity,'load_profile':'uncontrolled shared services; job='+key})
        common=['--data',str(inp),'--model-path',str(model),'--output',str(out),'--identity',str(ident),
                '--device','cuda:0','--seed','7','--allow-test','--bbeh-scorer',scorer,
                '--offload-kv-cache','--max-gpu-memory-gib','8','--max-context','131072','--load-in-4bit']
        if training and training.name.startswith('craft_quality'):
            argv=['comparison_craft.py',*common,'--training-run',str(training)]
        else:
            argv=['comparison_suite.py','run',*common,'--method',str(repo/'experiments/comparison/methods'/(method+'.json'))]
            if training:argv+=['--adapter',str(training/'adapter-final')]
            if released:
                argv.remove('--load-in-4bit')
                argv[argv.index('--model-path')+1]=str(root/'models/thinkless-1.5b-rl')
                argv+=['--loader','causal']
                argv[argv.index('--max-context')+1]='32768'
        add(key,argv,out,'eval');evals[cohort][label]=out
    # First full CRAFT run, then primary math results; all training arms and
    # evaluation settings remain frozen regardless of the first observed scores.
    for seed in SEEDS:
        for algorithm in ALGORITHMS:
            label=f'{algorithm}-s{seed}'
            original=json.loads((repo/'experiments/comparison/training'/(algorithm+'.json')).read_text())
            cfg=TrainConfig(**{**original,'seed':seed,'load_in_4bit':True,'activation_offload':True,
                              'offload_kv_cache':False,'max_gpu_memory_gib':8.,'checkpoint_every':5})
            config=campaign/'configs'/(label+'.json');write_json(config,asdict(cfg))
            out=campaign/'train'/label
            add(label,['comparison_training.py','--data',str(data/'train.jsonl'),'--config',str(config),
                       '--model-path',str(model),'--device','cuda:0','--output',str(out)],out,'train')
            if seed==7 and algorithm=='craft_quality':
                evaluation(label,'math',out)
                evaluation('base-cot','math')
    for method in ['direct','cot','cod','plan_solve','self_verify','self_consistency_k3','self_rank_k3','self_refine','native_thinking']:
        if method!='cot':evaluation('base-'+method,'math',method=method)
    evaluation('released-thinkless-common-cot','math',released=True)
    for seed in SEEDS:
        for algorithm in ALGORITHMS:
            label=f'{algorithm}-s{seed}'
            if label!='craft_quality-s7':evaluation(label,'math',campaign/'train'/label)
    for method in ['direct','cot','cod','self_consistency_k3']:
        evaluation('base-'+method,'legacy',method=method)
    for seed in SEEDS:
        for algorithm in ALGORITHMS:
            label=f'{algorithm}-s{seed}';evaluation(label,'legacy',campaign/'train'/label)
    for cohort,runs in evals.items():
        argv=['comparison_suite.py','compare','--baseline',str(runs['base-cot']),
              '--output',str(campaign/'comparisons'/(cohort+'.json')),'--replicates','10000']
        for label,out in runs.items():
            if label!='base-cot':argv+=['--candidate',str(out)]
        add('compare-'+cohort,argv,campaign/'comparisons'/(cohort+'.json'),'compare')
    inputs={}
    for job in jobs:
        for flag in ['--data','--config','--identity','--method','--bbeh-scorer']:
            if flag in job['argv']:
                path=job['argv'][job['argv'].index(flag)+1];inputs[path]=file_hash(path)
    plan={'schema':1,'input_files_sha256':inputs,'jobs':jobs,'repo':str(repo),'python':str(Path(sys.executable)),
          'source_hashes':source_hashes(),'data_audit_sha256':file_hash(data/'audit.json'),
          'seeds':SEEDS,'inference_seed':7,'evaluation_note':'full cohorts; shared inference random seed across independent training seeds',
          'training_note':'complete registered 200-step schedules or 1M-token ceiling; not a full data epoch or equal realized compute',
          'resource':{'physical_gpu':1,'allocator_gib':8,'min_free_start_mib':9216,'stop_services':False},
          'limitations':['reference trainers are not full upstream reproductions','released Thinkless has a different base and training set',
                        'latencies have uncontrolled service contention','no claim of SOTA from this campaign alone']}
    write_json(campaign/'plan.json',plan)
    print(json.dumps({'plan':str(campaign/'plan.json'),'jobs':len(jobs),'digest':digest(plan)}))


def execute(plan_path):
    plan_path=Path(plan_path).resolve();root=plan_path.parent;plan=json.loads(plan_path.read_text())
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if plan['source_hashes']!=source_hashes():raise ValueError('campaign source changed after freeze')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    status={'plan_digest':digest(plan),'worker_pid':os.getpid(),'status':'running','completed':[]}
    status_path=root/'status.json'
    for job in plan['jobs']:
        status.update(current_job=job['name'],job_started_at=time.time());atomic_json(status_path,status)
        if plan['source_hashes']!=source_hashes():raise ValueError('campaign code changed while running')
        for flag in ['--data','--config','--identity','--method','--bbeh-scorer']:
            if flag in job['argv']:
                path=job['argv'][job['argv'].index(flag)+1]
                if file_hash(path)!=plan['input_files_sha256'][path]:raise ValueError('frozen campaign input changed')
        out=Path(job['output']);argv=[plan['python'],*job['argv']]
        if job['kind']=='compare' and out.exists():
            # Compare output is immutable; successful prior execution is checked
            # against its recorded zero exit status before skipping.
            marker=root/'logs'/(job['name']+'.exit.json')
            if not marker.exists() or json.loads(marker.read_text())['returncode']!=0:raise ValueError('unverified comparison output')
            status['completed'].append(job['name']);continue
        if job['kind']!='compare':
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=plan['resource']['min_free_start_mib']:break
                status.update(status='waiting-for-free-memory',free_mib=free);atomic_json(status_path,status);time.sleep(30)
            status['status']='running';atomic_json(status_path,status)
            if out.exists():argv.append('--resume')
        log=root/'logs'/(job['name']+'.log')
        with log.open('a') as stream:
            stream.write('\nINVOCATION '+json.dumps({'argv':argv,'time':time.time()})+'\n');stream.flush()
            result=subprocess.run(argv,cwd=plan['repo'],env=env,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(root/'logs'/(job['name']+'.exit.json'),{'returncode':result.returncode,'time':time.time()})
        if result.returncode:
            status.update(status='failed',returncode=result.returncode,log=str(log));atomic_json(status_path,status)
            return result.returncode
        status['completed'].append(job['name']);atomic_json(status_path,status)
        from comparison_report import report
        report(root)
    status.update(status='complete',completed_at=time.time());atomic_json(status_path,status)
    from comparison_report import report
    report(root)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    b=subs.add_parser('build');b.add_argument('--root',required=True);b.add_argument('--model',required=True)
    r=subs.add_parser('execute');r.add_argument('--plan',required=True)
    a=p.parse_args()
    if a.command=='build':build(a.root,a.model)
    else:raise SystemExit(execute(a.plan))
