"""Freeze full official cohorts from pinned local exports; no model selection."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from comparison_data import (Scorer, assert_disjoint, convert, digest, file_hash,
                             read_rows, split_training, validate_cases, write_json)
from comparison_suite import export_legacy, write_cases

REVISIONS = {'gsm8k':'3101c7d5072418e28b9008a6636bde82a006892c',
             'math':'21a5633873b6a120296cce3e2df9d5550074f4a3',
             'math500':'6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'}
# Official training solutions have empty boxed answers. Exclude before splitting,
# without guessing replacement labels. Evaluation cohorts have no exclusions.
INVALID_MATH = {
 'number_theory/661':'66d3a1871c21837c0a0329683af05267227e69884fd1791530025f83cea5ddf8',
 'number_theory/663':'65443d7c0ee43ac5426f7bde2ad4fc5b8c3590a1df21dd3c8dbaaba6e90ad55e'}


def prepare(sources, output, legacy_root):
    sources,output=Path(sources),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    math=read_rows(sources/'math-train.jsonl')
    assert len(math)==7500
    excluded=[];clean=[]
    for row in math:
        if row['id'] in INVALID_MATH:
            if digest(row)!=INVALID_MATH[row['id']]:raise ValueError('excluded source row changed')
            excluded.append({'id':row['id'],'sha256':digest(row),'reason':'empty official boxed gold'})
        else:clean.append(row)
    if len(excluded)!=2:raise ValueError('unexpected training data defects')
    cleaned=output/'math-clean-source.jsonl'
    with cleaned.open('x') as f:
        for row in clean:f.write(json.dumps(row)+'\n')
    specs=[('gsm8k','train',sources/'gsm8k/grade_school_math/data/train.jsonl',7473),
           ('gsm8k','test',sources/'gsm8k/grade_school_math/data/test.jsonl',1319),
           ('math','train',cleaned,7498),('math500','test',sources/'math500-test.jsonl',500)]
    train=[];tests={};source_hashes={}
    for dataset,split,path,count in specs:
        cases=convert(path,dataset,dataset,split,split,REVISIONS[dataset])
        # Upstream row indices restart in each split.
        cases=[replace(c,id=c.dataset+'/'+split+'/'+c.id[len(c.dataset)+1:]) for c in cases]
        if len(cases)!=count:raise ValueError('source count mismatch')
        Scorer(cases)
        source_hashes[f'{dataset}-{split}']=file_hash(path)
        if split=='train':train.extend(cases)
        else:tests[dataset]=cases
    held=[c for rs in tests.values() for c in rs]
    # Any upstream train/test overlap must be audited, never silently retained.
    assert_disjoint(train,held)
    divided=split_training(train,.1,7)
    roles={role:[c for c in divided if c.split==role] for role in ('train','dev')}
    assert_disjoint(roles['train'],roles['dev'])
    provenance={'revisions':REVISIONS,'source_hashes':source_hashes,'excluded_training':excluded,
                'split':'10% whole question groups; seed 7; no evaluation filtering'}
    manifests={}
    for role,cases in roles.items():manifests[role]=write_cases(output/(role+'.jsonl'),cases,provenance)
    for name,cases in tests.items():manifests[name]=write_cases(output/(name+'-test.jsonl'),cases,provenance)
    for split in ('validation','test'):
        cases=export_legacy(legacy_root,['bbh','bbeh','usr'],split)
        assert_disjoint(roles['train'],cases)
        manifests['legacy-'+split]=write_cases(output/('legacy-'+split+'.jsonl'),cases,
                      {'source':'existing frozen repo benchmark splits','split':split,'no_filtering':True})
    # All legacy held-out rows are transfer evaluation, including the old dev
    # slice. Keep its split identity in the audit but do not use it for training.
    combined=held+export_legacy(legacy_root,['bbh','bbeh','usr'],'test')
    manifests['primary-test']=write_cases(output/'primary-test.jsonl',combined,provenance)
    write_json(output/'audit.json',{'manifests':manifests,'provenance':provenance})
    print(json.dumps({k:v['n'] for k,v in manifests.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--legacy-root',required=True,type=Path)
    a=p.parse_args();prepare(a.sources,a.output,a.legacy_root)
