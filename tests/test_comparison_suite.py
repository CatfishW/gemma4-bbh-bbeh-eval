import copy
from dataclasses import asdict,replace
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import comparison_suite as cs


def case(i=0,**kw):
    values=dict(id=f'd/{i}',dataset='d',task='task',group=f'g/{i}',question=f'question-{i}',
                target='42',split='dev',source_split='train',revision='abc',scorer='numeric')
    values.update(kw);return cs.Case(**values)


class FakeBackend:
    def __init__(self,texts=None,tokens=2):self.texts=list(texts or ['Final answer: 42']*100);self.seen=[];self.tokens=tokens
    def generate(self,prompt,cap,temperature,seed,thinking=False):
        self.seen.append((prompt,cap,temperature,seed,thinking))
        response=self.texts.pop(0)
        if isinstance(response,Exception):raise response
        return cs.Generation(response,min(self.tokens,cap) if self.tokens is not None else None,10,.01,'stop')


class DataTests(unittest.TestCase):
    def convert(self,rows,fmt,**kw):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.json';p.write_text(json.dumps(rows))
            return cs.convert(p,fmt,fmt,'test','test','rev',**kw)

    def test_gsm8k_reference(self):
        c=self.convert([{'question':'q','answer':'work #### 1,234'}],'gsm8k')[0]
        self.assertEqual(c.target,'1,234');self.assertEqual(c.scorer,'numeric')

    def test_missing_gsm_reference_rejected(self):
        with self.assertRaises(ValueError):self.convert([{'question':'q','answer':'unstructured 5'}],'gsm8k')

    def test_math_nested_box(self):
        c=self.convert([{'problem':'q','solution':r'thus \boxed{\frac{1}{2}}'}],'math')[0]
        self.assertEqual(c.target,r'\frac{1}{2}')

    def test_math500_and_aime(self):
        self.assertEqual(self.convert([{'problem':'q','answer':'2'}],'math500')[0].scorer,'math_verify')
        self.assertEqual(self.convert([{'Problem':'q','Answer':42}],'aime')[0].target,'42')

    def test_no_unstructured_math_gold(self):
        with self.assertRaises(ValueError):self.convert([{'problem':'q','solution':'the answer may be 3'}],'math')

    def test_gpqa_shuffle_is_deterministic_and_maps_gold(self):
        raw=[{'Question':f'q{i}','Correct Answer':'right','Incorrect Answer 1':'a','Incorrect Answer 2':'b','Incorrect Answer 3':'c'} for i in range(20)]
        one=self.convert(raw,'gpqa');two=self.convert(raw,'gpqa')
        self.assertEqual(one,two);self.assertGreater(len({c.target for c in one}),1)
        self.assertTrue(all(c.choices[ord(c.target)-65]=='right' for c in one))

    def test_mmlu_pro_numeric_reference(self):
        c=self.convert([{'question':'q','options':['a','b','c'],'answer_index':2,'category':'law'}],'mmlu_pro')[0]
        self.assertEqual(c.target,'C');self.assertEqual(c.task,'law')

    def test_svamp_and_deepscaler(self):
        self.assertEqual(self.convert([{'Body':'3 apples.','Question':'count?','Answer':3}],'svamp')[0].target,'3')
        self.assertEqual(self.convert([{'problem':'q','answer':'\\sqrt{2}'}],'deepscaler')[0].scorer,'math_verify')

    def test_official_test_cannot_be_training(self):
        with self.assertRaises(ValueError):case(split='train',source_split='test')

    def test_group_split_has_no_crossing(self):
        rows=[case(i,group=f'g/{i//2}',split='train') for i in range(20)]
        split=cs.split_training(rows,.2,7)
        self.assertEqual(len([c for c in split if c.split=='dev']),4)
        cs.assert_disjoint([c for c in split if c.split=='train'],[c for c in split if c.split=='dev'])

    def test_only_train_can_be_subdivided(self):
        with self.assertRaises(ValueError):cs.split_training([case()])

    def test_explicit_group_and_question_leakage(self):
        for c in (case(1,group='g/0'),case(1,question='question-0')):
            with self.assertRaises(ValueError):cs.assert_disjoint([case()],[c])

    def test_loader_rejects_duplicate_and_split_crossing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl'
            for rows in ([case(),case()],[case(),case(1,group='g/0',split='train')]):
                p.write_text(''.join(json.dumps(asdict(c))+'\n' for c in rows))
                with self.assertRaises(ValueError):cs.load_cases(p)

    def test_explicit_expected_count(self):
        with tempfile.TemporaryDirectory() as d,patch('sys.stderr',new_callable=io.StringIO):
            root=Path(d);p=root/'gsm.json';p.write_text('[{"question":"q","answer":"#### 5"}]')
            with self.assertRaises(SystemExit):cs.main(['prepare','--input',str(p),'--output',str(root/'out'),
                '--format','gsm8k','--dataset','gsm8k','--split','test','--source-split','test','--revision','pinned','--expected-count','2'])
            self.assertFalse((root/'out').exists())


class ScoringTests(unittest.TestCase):
    def test_no_arbitrary_last_number(self):
        self.assertIsNone(cs.numeric('I considered 42 and 57'))
        self.assertIsNone(cs.numeric('There are 42 apples'))

    def test_exact_numeric_without_floats(self):
        for txt in ('#### 0.5',r'\boxed{1/2}','Final answer: .5'):
            self.assertEqual(cs.numeric(txt),Fraction(1,2))
        self.assertEqual(cs.numeric('1,234,567'),1234567)
        self.assertIsNone(cs.numeric('1,23'));self.assertIsNone(cs.numeric('1/0'))

    def test_xml_box_markers(self):
        self.assertEqual(cs.final_text('work <answer>A</answer>'),'A')
        self.assertEqual(cs.final_text('Final answer: B\nextra'),'B')
        self.assertEqual(cs.final_text(r'\boxed{1} then \boxed{\frac{2}{3}}'),r'\frac{2}{3}')

    def test_choice_only_no_guessing(self):
        c=case(scorer='choice',target='B',choices=('red','blue'))
        s=cs.Scorer([c]);self.assertTrue(s('Final answer: (B)',c));self.assertFalse(s('A or B',c));self.assertFalse(s('blue',c))

    def test_empty_wrong_and_math_library_explicit(self):
        c=case();self.assertFalse(cs.Scorer([c])('',c))
        with patch.dict('sys.modules',{'math_verify':None}),self.assertRaises(ImportError):cs.Scorer([case(scorer='math_verify')])

    def test_bbeh_requires_pinned_official_scorer(self):
        with self.assertRaises(ValueError):cs.Scorer([case(scorer='bbeh_official')])


class MethodTests(unittest.TestCase):
    def test_direct_one_call_and_no_target(self):
        b=FakeBackend();r=cs.run_method('private question',cs.Method(name='direct'),b,7)
        self.assertEqual(len(b.seen),1);self.assertNotIn('gold-secret',b.seen[0][0]);self.assertEqual(r['output_tokens'],2)

    def test_cot_cod_plan_native(self):
        for name in ('cot','cod','plan_solve','native_thinking','self_verify'):
            b=FakeBackend();cs.run_method('q',cs.Method(name=name),b,7)
            self.assertEqual(b.seen[0][-1],name=='native_thinking')
        self.assertIn('5 words at most',cs.SUFFIXES['cod'])

    def test_majority_not_concatenated_prose(self):
        b=FakeBackend(['#### 42','Final answer: 41','Final answer: 42'])
        r=cs.run_method('q',cs.Method(name='self_consistency',budget=15),b,7)
        self.assertEqual(cs.final_text(r['prediction']),'42');self.assertEqual(len(r['calls']),3)

    def test_empty_votes_never_win(self):
        r=cs.run_method('q',cs.Method(name='self_consistency'),FakeBackend(['','','#### 42']),7)
        self.assertEqual(cs.final_text(r['prediction']),'42')

    def test_self_rank_reserves_final_call(self):
        b=FakeBackend(['A','B','B','B']);r=cs.run_method('q',cs.Method(name='self_rank',budget=40,final_budget=10),b,7)
        self.assertEqual([s[1] for s in b.seen],[10]*4);self.assertEqual(b.seen[-1][2],0);self.assertEqual(r['prediction'],'B')

    def test_self_refine_full_feedback_cost(self):
        b=FakeBackend(['A','contradiction','B']);r=cs.run_method('q',cs.Method(name='self_refine',budget=9),b,7)
        self.assertEqual(len(r['calls']),3);self.assertEqual(r['prediction'],'B');self.assertEqual(r['output_tokens'],6)

    def test_unknown_cost_is_not_zero_and_is_reserved(self):
        r=cs.run_method('q',cs.Method(name='self_consistency',budget=9),FakeBackend(tokens=None),7)
        self.assertIsNone(r['output_tokens']);self.assertEqual(r['charged_completion_tokens'],9)

    def test_failure_no_retry_or_label_trigger(self):
        b=FakeBackend([TimeoutError(),'A']);r=cs.run_method('q',cs.Method(name='self_consistency'),b,7)
        self.assertEqual(len(b.seen),1);self.assertTrue(r['error']);self.assertEqual(r['prediction'],'')

    def test_backend_overrun_is_rejected(self):
        class Bad:
            def generate(self,*args,**kw):return cs.Generation('A',999999,1,.1)
        with self.assertRaises(ValueError):cs.run_method('q',cs.Method(budget=2),Bad(),7)

    def test_invalid_budgets_and_fake_paper_names(self):
        for kw in ({'name':'tot'},{'name':'s1'},{'name':'self_rank','budget':128},{'name':'self_consistency','budget':1},{'temperature':float('nan')}):
            with self.assertRaises(ValueError):cs.Method(**kw)

    def test_seed_deterministic(self):
        a,b=FakeBackend(),FakeBackend();m=cs.Method(name='self_consistency')
        cs.run_method('q',m,a,7);cs.run_method('q',m,b,7);self.assertEqual(a.seen,b.seen)

    def test_api_preserves_channels_and_never_saves_key(self):
        payload={'choices':[{'message':{'content':'#### 42','reasoning_content':'thinking'},'finish_reason':'stop'}],
                 'usage':{'prompt_tokens':3,'completion_tokens':6,'completion_tokens_details':{'reasoning_tokens':4}}}
        sent=[]
        def fake(req,timeout):sent.append(json.loads(req.data));return io.BytesIO(json.dumps(payload).encode())
        with patch('comparison_suite.request.urlopen',fake),patch.dict('os.environ',{'OPENAI_API_KEY':'gold-secret'}):
            r=cs.APIBackend('http://localhost/v1','test').generate('q',8,0,7)
        self.assertNotIn('gold-secret',json.dumps(asdict(r)));self.assertEqual(r.output_tokens,6)
        self.assertEqual(sent[0]['messages'],[{'role':'user','content':'q'}]);self.assertFalse(sent[0]['chat_template_kwargs']['enable_thinking'])

    def test_api_endpoint_secrets_rejected(self):
        for url in ('file:///a','https://secret@host/v1','https://host?key=x'):
            with self.assertRaises(ValueError):cs.APIBackend(url,'m')


def identity():
    return dict(base_model_sha256='a'*64,checkpoint_sha256='b'*64,hardware_id='cpu-test',precision='FP32',serving_stack='fake',load_profile='single',study_id='unit-test')


class StatsAndArtifactTests(unittest.TestCase):
    def rows(self):
        return [dict(id=str(i),case_digest=str(i),group=f'g/{i//2}',dataset='d',task=f't/{i//4}',correct=i%2==0,
                     elapsed_seconds=1.,error=None,input_tokens=20,output_tokens=2,calls=[]) for i in range(8)]

    def test_exact_mcnemar_and_holm(self):
        self.assertEqual(cs.mcnemar(0,0),1);self.assertAlmostEqual(cs.mcnemar(5,0),.0625)
        self.assertEqual(cs.holm([.01,.04,.03]),[.03,.06,.06]);self.assertTrue(0<=cs.mcnemar(1100,238)<=1)

    def test_bootstrap_identical(self):
        rows=self.rows();r=cs.paired(rows,rows,reps=100)
        self.assertEqual(r['accuracy_ci95'],[0,0]);self.assertEqual(r['latency_ratio_ci95'],[1,1]);self.assertEqual(r['groups'],4)

    def test_bootstrap_task_clusters(self):
        r=cs.paired(self.rows(),self.rows(),reps=100,cluster='task');self.assertEqual(r['groups'],2)

    def test_pair_coverage_and_case_identity(self):
        a=self.rows();b=copy.deepcopy(a);b[0]['case_digest']='changed'
        with self.assertRaises(ValueError):cs.paired(a,b,reps=100)
        with self.assertRaises(ValueError):cs.paired(a,a[:-1],reps=100)

    def test_missing_usage_remains_missing(self):
        r=self.rows();r[0]['output_tokens']=None;self.assertIsNone(cs.summarize(r)['mean_output_tokens'])

    def test_gold_not_sent_and_complete_digest(self):
        with tempfile.TemporaryDirectory() as d:
            c=case(target='gold-secret',scorer='exact');b=FakeBackend(['answer'])
            cs.run([c],cs.Method(name='direct'),b,Path(d)/'run',identity(),cs.Scorer([c]))
            self.assertNotIn('gold-secret',b.seen[0][0]);cfg,rows=cs.completed(Path(d)/'run');self.assertEqual(len(rows),1)
            p=Path(d)/'run/predictions.jsonl';p.write_text(p.read_text().replace('answer','tampered'))
            with self.assertRaises(ValueError):cs.completed(Path(d)/'run')

    def test_external_import_recomputes_correctness(self):
        with tempfile.TemporaryDirectory() as d:
            c=case();p=Path(d)/'source.jsonl'
            p.write_text(json.dumps({'id':c.id,'case_digest':cs.digest(asdict(c)),'prediction':'#### 41',
                                    'correct':True,'elapsed_seconds':1.,'output_tokens':2,'input_tokens':3})+'\n')
            meta={'method':'pal','upstream_commit':'a'*40,'implementation':'author-code','identity':identity(),'seed':7,'all_calls_accounted':False}
            report=cs.import_run([c],p,Path(d)/'out',meta,cs.Scorer([c]));self.assertEqual(report['accuracy'],0)
            self.assertIsNone(report['mean_output_tokens']);cs.completed(Path(d)/'out')

    def test_external_requires_exact_case_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.jsonl';p.write_text(json.dumps({'id':'d/0','prediction':'42'})+'\n')
            meta={'method':'pal','upstream_commit':'a'*40,'implementation':'author','identity':identity(),'seed':7,'all_calls_accounted':True}
            with self.assertRaises(ValueError):cs.import_run([case()],p,Path(d)/'out',meta,cs.Scorer([case()]))

    def test_partial_run_cannot_compare(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cs.write_json(p/'protocol.json',{});cs.write_json(p/'summary.json',{'status':'started'})
            (p/'predictions.jsonl').write_text('')
            with self.assertRaises(ValueError):cs.completed(p)


if __name__=='__main__':unittest.main()
