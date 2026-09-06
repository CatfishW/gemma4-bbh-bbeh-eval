"""Explicit acquisition of pinned public artifacts; never executed on import."""
import argparse
import json
from pathlib import Path
import subprocess

from comparison_prepare_campaign import REVISIONS
from comparison_data import file_hash, write_json

THINKLESS_REVISION='73551801776c64ecd599c46229120daae4934aec'


def acquire(root,thinkless=False):
    from huggingface_hub import hf_hub_download, snapshot_download
    import pyarrow.parquet as pq
    root=Path(root);sources=root/'data-sources';sources.mkdir(parents=True,exist_ok=True)
    gsm=sources/'gsm8k'
    if not gsm.exists():
        subprocess.run(['git','clone','https://github.com/openai/grade-school-math.git',str(gsm)],check=True)
        subprocess.run(['git','-C',str(gsm),'checkout','--detach',REVISIONS['gsm8k']],check=True)
    revision=subprocess.check_output(['git','-C',str(gsm),'rev-parse','HEAD'],text=True).strip()
    if revision!=REVISIONS['gsm8k']:raise ValueError('GSM8K checkout revision mismatch')
    rows=[]
    for kind in ['algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus']:
        path=hf_hub_download('EleutherAI/hendrycks_math',kind+'/train-00000-of-00001.parquet',repo_type='dataset',revision=REVISIONS['math'])
        records=pq.read_table(path).to_pylist()
        for i,r in enumerate(records):r['id']=kind+'/'+str(i)
        rows.extend(records)
    if len(rows)!=7500:raise ValueError('MATH training count changed')
    def same_or_create(path,content):
        if path.exists():
            if path.read_bytes()!=content:raise ValueError('existing source export differs')
        else:path.write_bytes(content)
    same_or_create(sources/'math-train.jsonl',(''.join(json.dumps(r)+'\n' for r in rows)).encode())
    path=hf_hub_download('HuggingFaceH4/MATH-500','test.jsonl',repo_type='dataset',revision=REVISIONS['math500'])
    same_or_create(sources/'math500-test.jsonl',Path(path).read_bytes())
    if thinkless:
        path=snapshot_download('Vinnnf/Thinkless-1.5B-RL-DeepScaleR',revision=THINKLESS_REVISION,
            local_dir=root/'models/thinkless-1.5b-rl',allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','README.md'])
        print('Released Thinkless checkpoint:',path)
    print('Pinned official data exports:',sources)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--thinkless',action='store_true');a=p.parse_args();acquire(a.root,a.thinkless)
