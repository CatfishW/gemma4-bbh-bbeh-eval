"""Versioned dataset conversion and scoring for augmentation comparisons.

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


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path) -> list[dict]:
    path = Path(path)
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.json':
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get('examples', value.get('data'))
        if not isinstance(value, list):
            raise ValueError('JSON dataset must be a list or an examples/data object')
        return value
    return [json.loads(line) for line in text.split('\n') if line.strip()]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


SCORERS = {'exact', 'numeric', 'choice', 'math_verify', 'legacy', 'bbeh_official'}


@dataclass(frozen=True)
class Case:
    id: str
    dataset: str
    task: str
    group: str
    question: str
    target: str
    split: str
    source_split: str
    revision: str
    scorer: str
    choices: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ('id','dataset','task','group','question','target','source_split','revision'):
            if not isinstance(getattr(self,key), str) or not getattr(self,key).strip():
                raise ValueError(f'nonempty {key} required')
        if self.source_split not in {'train','validation','test','evaluation-only'}:
            raise ValueError('unknown source split')
        if self.split not in {'train','dev','test'} or self.scorer not in SCORERS:
            raise ValueError('unsupported split/scorer')
        if self.source_split in {'test','validation','evaluation-only'} and self.split == 'train':
            raise ValueError('official held-out sources cannot be relabeled as training')
        if self.choices and (len(self.choices)>26 or any(not isinstance(c,str) or not c for c in self.choices)):
            raise ValueError('invalid choices')
        if self.scorer == 'choice' and (not self.choices or self.target not in [chr(65+i) for i in range(len(self.choices))]):
            raise ValueError('choice reference outside declared options')


def load_cases(path) -> list[Case]:
    rows = [Case(**{**r, 'choices':tuple(r.get('choices',[]))}) for r in read_rows(path)]
    return validate_cases(rows)


def validate_cases(rows):
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError('empty/duplicate case identities')
    groups, questions = {}, {}
    for r in rows:
        for lookup, key in ((groups, r.group), (questions, digest(r.question.strip()))):
            if key in lookup and lookup[key] != r.split:
                raise ValueError('related group or exact duplicate question crosses splits')
            lookup[key] = r.split
    return rows


def final_text(text: str) -> str:
    """Deterministic final-answer extraction; never guess the last arbitrary number."""
    tags = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.I | re.S)
    if tags:
        return tags[-1].strip()
    # TeX also permits a single unbraced atom (two official MATH rows).
    bare = list(re.finditer(r'\\boxed\s+([0-9A-Za-z])(?=[$\s.,;]|$)', text))
    starts = [m.end() for m in re.finditer(r'\\boxed\{', text)]
    if bare and (not starts or bare[-1].start()>starts[-1]):
        return bare[-1].group(1)
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                return text[start:i].strip()
    markers = list(re.finditer(r'####\s*|(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*:\s*', text, re.I))
    if markers:
        tail=text[markers[-1].end():].strip().splitlines()
        return tail[0].strip() if tail else ''
    return text.strip()


def vote_key(text):
    return ' '.join(final_text(text).strip(' \t\n\r.').split()).casefold()


def numeric(text):
    value = final_text(text).strip().strip('$')
    # Group separators allowed only when valid; units and arbitrary trailing prose are not.
    value = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', value)
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


class Scorer:
    def __init__(self, cases, bbeh_scorer=None):
        self.official = None
        names = sorted({c.scorer for c in cases})
        self.identity = {'names':names, 'suite_sha256':file_hash(__file__)}
        if 'math_verify' in names:
            from math_verify import parse as mp, verify
            self.mp, self.mv = mp, verify
            self.identity['math_verify'] = importlib.metadata.version('math-verify')
            self.identity['latex2sympy2_extended'] = importlib.metadata.version('latex2sympy2_extended')
            # Check references before generating any responses, not only after an expensive run.
            for c in cases:
                if c.scorer == 'math_verify' and not mp('$'+c.target.strip('$')+'$'):
                    raise ValueError(f'unparseable gold math reference: {c.id}')
        if 'legacy' in names:
            import eval_benchmarks as legacy
            self.legacy = legacy.evaluate_correctness
            self.identity['legacy_sha256'] = file_hash(legacy.__file__)
        if 'bbeh_official' in names:
            if bbeh_scorer is None:
                raise ValueError('supply the pinned upstream bbeh/evaluate.py via --bbeh-scorer')
            spec = importlib.util.spec_from_file_location('comparison_bbeh_official', bbeh_scorer)
            if spec is None or spec.loader is None:
                raise ValueError('invalid BBEH scorer module')
            module = importlib.util.module_from_spec(spec)
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
            self.official = module.evaluate_correctness
            self.identity['bbeh_sha256'] = file_hash(bbeh_scorer)

    def __call__(self, text, case):
        if not text.strip():
            return False
        if case.scorer == 'numeric':
            a,b = numeric(text), numeric(case.target)
            return a is not None and b is not None and a == b
        if case.scorer == 'choice':
            value = final_text(text).strip().rstrip('.').strip('()').strip()
            return value.upper() == case.target if re.fullmatch('[A-Za-z]',value) else False
        if case.scorer == 'math_verify':
            pred = self.mp(text)
            return bool(pred) and bool(self.mv(self.mp('$'+case.target.strip('$')+'$'),pred))
        if case.scorer == 'legacy':
            return bool(self.legacy(text,case.target))
        if case.scorer == 'bbeh_official':
            return bool(self.official(text,case.target))
        return vote_key(text) == vote_key(case.target)


def convert(path, fmt, dataset, split, source_split, revision, scorer=None, group_field=None, seed=7):
    """Read local official exports; never fetch online or silently shorten a dataset."""
    raw = read_rows(path)
    cases = []
    for index, r in enumerate(raw):
        options = []
        task = str(r.get('task',r.get('category',r.get('type',dataset))))
        rid = str(r.get('id',r.get('question_id',index)))
        if fmt == 'gsm8k':
            q = r['question']; target = str(r['answer']).rsplit('####',1)[-1].strip(); default = 'numeric'
            if '####' not in str(r['answer']):
                raise ValueError('GSM8K reference missing ####')
        elif fmt in {'math','math500','aime','deepscaler'}:
            q = r.get('problem',r.get('question',r.get('Problem')))
            ref = r.get('answer',r.get('Answer',r.get('solution')))
            if ref is None: raise ValueError('missing math answer/solution')
            target = final_text(str(ref)); default = 'numeric' if fmt=='aime' else 'math_verify'
            if fmt=='math' and 'answer' not in r and not re.search(r'\\boxed(?:\{|\s+[0-9A-Za-z](?=[$\s.,;]|$))',str(ref)):
                raise ValueError('MATH solution has no boxed final reference')
        elif fmt == 'gpqa':
            q = r['Question']; options = [r['Correct Answer'],*[r[f'Incorrect Answer {i}'] for i in (1,2,3)]]
            order = list(range(4)); random.Random(digest([seed,dataset,rid,q])).shuffle(order)
            options = [options[j] for j in order]; target = chr(65+order.index(0)); default='choice'
        elif fmt == 'svamp':
            q = str(r.get('Body',''))+' '+r['Question']; target=str(r['Answer']);default='numeric'
        elif fmt == 'mmlu_pro':
            q = r['question']; options = r['options']
            if not isinstance(options,list): raise ValueError('options must be a JSON list')
            ref = r.get('answer',r.get('answer_index'))
            target = chr(65+ref) if type(ref) is int else str(ref); default='choice'
        elif fmt in {'bbh','bbeh'}:
            q,target = r['input'],str(r['target']); default = 'legacy' if fmt=='bbh' else 'bbeh_official'
        elif fmt == 'canonical':
            q,target = r['question'],str(r['target']); options = r.get('choices',[]); default=r.get('scorer','exact')
            task=str(r.get('task',dataset))
        else:
            raise ValueError('unsupported converter')
        if not isinstance(q,str) or not q.strip(): raise ValueError('invalid question')
        if options: q += '\n'+'\n'.join(f'({chr(65+i)}) {v}' for i,v in enumerate(options))
        if group_field and group_field not in r: raise ValueError('declared group field missing')
        # Explicit group ids let puzzle variants/templated families stay together.
        group = str(r[group_field]) if group_field else str(r.get('group',digest(q.strip())))
        cases.append(Case(f'{dataset}/{rid}',dataset,task,f'{dataset}/{group}',q,target,split,
                          source_split,revision,scorer or default,tuple(options)))
    if not cases or len({c.id for c in cases})!=len(cases): raise ValueError('empty/duplicate converted ids')
    return cases


def split_training(cases, fraction=.1, seed=7):
    if not 0<fraction<1 or any(c.source_split!='train' or c.split!='train' for c in cases):
        raise ValueError('only official training data can be split for model development')
    from dataclasses import replace
    groups = sorted({c.group for c in cases})
    if len(groups)<2: raise ValueError('at least two independent groups required')
    random.Random(seed).shuffle(groups)
    held = set(groups[:max(1,min(len(groups)-1,round(len(groups)*fraction)))])
    return [replace(c,split='dev' if c.group in held else 'train') for c in cases]


def assert_disjoint(train, evaluation):
    if {c.id for c in train}&{c.id for c in evaluation}: raise ValueError('identity leakage')
    if {c.group for c in train}&{c.group for c in evaluation}: raise ValueError('group leakage')
    if {digest(c.question.strip()) for c in train}&{digest(c.question.strip()) for c in evaluation}:
        raise ValueError('question leakage')
