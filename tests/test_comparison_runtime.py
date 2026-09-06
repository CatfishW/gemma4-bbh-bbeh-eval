"""Recovery checks exercise interrupted work, integrity, and exact next updates."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import comparison_runtime as cr
import comparison_suite as cs
import test_comparison_training as fixtures
torch=fixtures.torch


class EvaluationRecoveryTests(unittest.TestCase):
    def test_resume_uses_only_verified_prefix(self):
        c=cs.Case('1','d','t','g','q','42','test','test','v1','numeric')
        protocol={'data':cs.digest(asdict(c))}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            self.assertEqual(cr.resume_predictions(out,protocol,[c],False),[])
            row={'id':c.id,'case_digest':cs.digest(asdict(c)),'correct':True}
            (out/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(cr.resume_predictions(out,protocol,[c],True),[row])
            with self.assertRaises(ValueError):cr.resume_predictions(out,{'data':'changed'},[c],True)
            row['case_digest']='modified'
            (out/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):cr.resume_predictions(out,protocol,[c],True)

    def test_jsonl_keeps_unicode_line_separators_inside_strings(self):
        from comparison_data import read_rows
        row={'question':'a\u2028b\u0085c'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl'
            path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
            self.assertEqual(read_rows(path),[row])

    def test_unbraced_official_math_answer(self):
        self.assertEqual(cs.final_text(r'Therefore $\boxed 2$.'),'2')
        self.assertEqual(cs.final_text(r'$\boxed{3}$ then $\boxed 9$'),'9')
        self.assertEqual(cs.final_text(r'$\boxed 9$ then $\boxed{3}$'),'3')


@unittest.skipIf(torch is None,'requires CPU torch')
class TrainingRecoveryTests(unittest.TestCase):
    def test_optimizer_rng_resume_reproduces_next_update(self):
        import comparison_training as ct
        fixture=fixtures.CPUTrainingTests();cases=fixture.cases();cfg=ct.TrainConfig(max_tokens=1)
        b=fixture.backend();engine=ct.ContentTrainer(b,cases,cfg,cs.Scorer(cases))
        engine.step()
        with patch.object(torch.cuda,'is_available',return_value=False):state=copy.deepcopy(cr.training_state(engine))
        weights=copy.deepcopy(b.model.state_dict())
        expected=engine.step()
        restored=fixture.backend();restored.model.load_state_dict(weights)
        resumed=ct.ContentTrainer(restored,cases,cfg,cs.Scorer(cases));cr.restore_training_state(resumed,state)
        actual=resumed.step()
        for k in ('loss','sampled_tokens','train_accuracy','gradient_norm'):
            self.assertEqual(expected[k],actual[k])
        for k,v in b.model.state_dict().items():self.assertTrue(torch.equal(v,restored.model.state_dict()[k]))

    def test_checkpoint_roundtrip_and_tamper(self):
        import comparison_training as ct
        fixture=fixtures.CPUTrainingTests();b=fixture.backend();cases=fixture.cases()
        def save(path):
            path.mkdir();torch.save(b.model.state_dict(),path/'weights.pt')
        b.save=save;engine=ct.ContentTrainer(b,cases,ct.TrainConfig(max_tokens=1),cs.Scorer(cases))
        engine.step();protocol={'fixture':'complete'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.cuda,'is_available',return_value=False):
            out=Path(tmp);cp=cr.save_training_checkpoint(engine,b,out,protocol,[])
            self.assertEqual(cr.latest_checkpoint(out,protocol),cp)
            state=torch.load(cp/'state.pt',weights_only=True)
            self.assertEqual(state['step'],1)
            cr.adapter_alias(out,'adapter-final',cp)
            self.assertEqual((out/'adapter-final').resolve(),cp/'adapter')
            with self.assertRaises(ValueError):cr.verify_training_checkpoint(cp,{'fixture':'changed'})
            (cp/'adapter/weights.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):cr.verify_training_checkpoint(cp,protocol)

class ResourceFailureTests(unittest.TestCase):
    def test_resource_failure_is_not_a_wrong_benchmark_answer(self):
        from comparison_methods import run_method, Method, BackendResourceError
        class Backend:
            def generate(self,*args,**kwargs):raise RuntimeError('CUDA out of memory')
        with self.assertRaises(BackendResourceError):run_method('q',Method(),Backend(),7)


@unittest.skipIf(torch is None,'requires CPU torch')
class AttentionTests(unittest.TestCase):
    def test_no_copy_decoder_matches_explicit_repetition(self):
        from rl_craft.attention import single_kv_decode
        generator=torch.Generator().manual_seed(13)
        q=torch.randn(1,8,1,16,generator=generator,dtype=torch.float64)
        k=torch.randn(1,1,37,16,generator=generator,dtype=torch.float64)
        v=torch.randn(1,1,37,16,generator=generator,dtype=torch.float64)
        for mask in (None,torch.cat([torch.ones(30),torch.zeros(7)]).bool().view(1,1,1,37)):
            actual,_=single_kv_decode(q,k,v,mask,scaling=.17)
            expected=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(8,1),v.repeat_interleave(8,1),attn_mask=mask,scale=.17)
            self.assertTrue(torch.allclose(actual,expected.transpose(1,2),atol=1e-12,rtol=1e-12))

class CampaignFailureTests(unittest.TestCase):
    def test_failed_job_stops_queue_and_only_selects_gpu_one(self):
        import comparison_campaign as campaign
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'logs').mkdir()
            plan={'source_hashes':campaign.source_hashes(),'repo':str(root),'python':'python',
                  'resource':{'min_free_start_mib':9216},'input_files_sha256':{},
                  'jobs':[{'name':name,'argv':['fixture.py'],'output':str(root/name),'kind':'train'} for name in ('a','b')]}
            (root/'plan.json').write_text(json.dumps(plan))
            with patch.object(campaign.subprocess,'check_output',return_value='10000'),patch.object(campaign.subprocess,'run',return_value=SimpleNamespace(returncode=3)) as run:
                self.assertEqual(campaign.execute(root/'plan.json'),3)
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'1')
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['status'],'failed');self.assertEqual(status['completed'],[])


class FinalAnswerTests(unittest.TestCase):
    def test_later_final_answer_overrides_earlier_box(self):
        self.assertEqual(cs.final_text(r'Intermediate: \boxed{42}. Final answer: 41'),'41')
        self.assertEqual(cs.final_text(r'Final answer: \boxed{42}'),'42')

    def test_bbeh_receives_common_final_extraction(self):
        c=cs.Case('id','bbeh','t','g','q','42','test','test','r1','bbeh_official')
        scorer=cs.Scorer.__new__(cs.Scorer);scorer.official=lambda pred,target:pred==target
        self.assertTrue(scorer('Reasoning goes here.\nFinal answer: 42',c))
        self.assertFalse(scorer('Reasoning goes here.\nFinal answer: 41',c))


@unittest.skipIf(__import__('importlib.util',fromlist=['find_spec']).find_spec('math_verify') is None,'requires math-verify')
class SymbolicScorerTests(unittest.TestCase):
    def test_math_gold_and_prediction_share_delimiters(self):
        pairs=[(r'p-q',r'p+q'),(r'3\sqrt{13}',r'3\sqrt{7}'),(r'\pi',r'2\pi'),
               (r'\left(3,\frac{\pi}{2}\right)',r'(3,-\pi/2)'),(r'\text{Evelyn}',r'\text{Adam}')]
        for right,wrong in pairs:
            c=cs.Case('id','math','t','g','q',right,'test','test','r1','math_verify')
            scorer=cs.Scorer([c])
            self.assertTrue(scorer(right,c))
            self.assertTrue(scorer('Final answer: '+right,c))
            self.assertFalse(scorer('Final answer: '+wrong,c))
