"""Publishable aggregates only; raw prompts, labels, token traces stay on host."""
import argparse
import json
from pathlib import Path
from comparison_data import digest, file_hash
from comparison_suite import completed


def report(campaign):
    root=Path(campaign);plan=json.loads((root/'plan.json').read_text())
    state=json.loads((root/'status.json').read_text())
    if state['plan_digest']!=digest(plan):raise ValueError('campaign plan mismatch')
    training=[];evaluations=[]
    for job in plan['jobs']:
        out=Path(job['output'])
        if job['kind']=='compare' or not (out/'summary.json').exists():continue
        summary=json.loads((out/'summary.json').read_text())
        if summary.get('status')!='complete':continue
        if job['kind']=='eval':
            protocol,rows=completed(out)
            evaluations.append({'run':job['name'],'summary':summary,'method':protocol['method'],
                                'data_digest':protocol['data_digest'],'identity':protocol['identity']})
        else:
            from comparison_runtime import latest_checkpoint
            protocol=json.loads((out/'protocol.json').read_text());latest_checkpoint(out,protocol)
            if summary['protocol_digest']!=digest(protocol):raise ValueError('training summary mismatch')
            training.append({'run':job['name'],'summary':summary,'config':protocol['config']})
    value={'status':state['status'],'plan_digest':digest(plan),'training':training,'evaluation':evaluations,
           'pending_jobs':[j['name'] for j in plan['jobs'] if j['name'] not in state['completed']],
           'sota_claim':False}
    lines=['# GPU 1 comparison campaign','',f"Status: **{state['status']}**. Completed {len(state['completed'])}/{len(plan['jobs'])} jobs.",'',
           'NF4 Gemma comparisons use an 8 GiB allocator limit on physical GPU 1. Existing services remain running. '
           'Wall-clock timing is descriptive because service contention is uncontrolled.','',
           'Training uses 13,474 official-source problems and a separate 1,497-problem development split. '
           'Two MATH training records with empty official answers were excluded before splitting. '
           'All 1,819 math test and 9,550 transfer test cases are retained.','',
           'Training seeds: 7, 17, 27. Inference seed: 7, shared across arms. '
           'Schedules stop at 200 updates or their 1,000,000-action-token ceiling; these are not full data epochs or equal realized compute.','',
           'RLOO/GRPO/Dr.GRPO are matched-runtime reference implementations. The released Thinkless checkpoint '
           'uses a different base and training set under the common CoT harness. These runs do not establish SOTA.','',
           '## Completed training','', '| Run | Steps | Action tokens | Stop reason |','|---|---:|---:|---|']
    for r in training:
        s=r['summary'];lines.append(f"| {r['run']} | {s['steps']} | {s['action_tokens']:,} | {s['stop_reason']} |")
    lines+=['','## Completed full-cohort evaluations','','| Run | Cases | Correct | Accuracy | Output tokens/case | Errors |','|---|---:|---:|---:|---:|---:|']
    for r in evaluations:
        s=r['summary'];tokens=s['mean_output_tokens'];token_text='unknown' if tokens is None else f'{tokens:.1f}'
        lines.append(f"| {r['run']} | {s['n']:,} | {s['correct']:,} | {s['accuracy']:.2%} | {token_text} | {s['errors']} |")
    if not evaluations:lines+=['','No full-cohort evaluation has completed yet.']
    lines+=['','The paired comparison JSON files are written only after all candidate cohorts finish. '
            'They include group/task bootstrap intervals, McNemar tests, Holm correction, and explicit cross-model labels.','']
    for name,content in [('aggregate.json',json.dumps(value,indent=2)+'\n'),('REPORT.md','\n'.join(lines))]:
        temp=root/(name+'.tmp');temp.write_text(content);temp.replace(root/name)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    value=report(a.campaign);print(json.dumps({'status':value['status'],'training':len(value['training']),'evaluation':len(value['evaluation'])}))
