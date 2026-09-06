from dataclasses import asdict,replace
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import comparison_suite as cs
import comparison_training as ct
try:import torch
except ImportError:torch=None
try:
    import rl_craft.core as craft_core
except ImportError:craft_core=None


class TrainingMathTests(unittest.TestCase):
    def test_quality_schedule(self):
        cfg=ct.TrainConfig(quality_warmup=3,cost_ramp=2,cost_weight=.2)
        self.assertEqual([ct.beta_at(i,cfg) for i in range(6)],[0,0,0,.1,.2,.2])

    def test_failed_lengths_indifferent(self):
        self.assertEqual(ct.utility(0,1,100,.2),ct.utility(0,10000,100,.2))
        self.assertEqual(ct.utility(0,100,100,0),0)

    def test_success_beats_failure_and_length_is_bounded(self):
        for beta in (0,.1,.99):
            self.assertGreater(ct.utility(1,100000,100,beta),ct.utility(0,1,100,beta))
            self.assertEqual(ct.utility(1,100000,100,beta),ct.utility(1,100,100,beta))

    def test_rloo_expected_group_values(self):
        got=ct.advantages([0,1,1,0],'rloo')
        for a,b in zip(got,[-2/3,2/3,2/3,-2/3]):self.assertAlmostEqual(a,b)

    def test_grpo_and_drgrpo_controls(self):
        self.assertEqual(ct.advantages([0,1],'drgrpo_reference'),[-.5,.5])
        self.assertAlmostEqual(ct.advantages([0,1],'grpo_reference')[0],-1,places=6)
        self.assertEqual(ct.advantages([1,1],'grpo_reference'),[0,0])

    def test_validation_fail_closed(self):
        for kwargs in ({'algorithm':'thinkless'},{'k':1},{'cost_weight':1},{'temperature':0},{'quality_warmup':-1}):
            with self.assertRaises(ValueError):ct.TrainConfig(**kwargs)
        with self.assertRaises(ValueError):ct.advantages([1],'rloo')


@unittest.skipIf(torch is None,'CPU torch not installed')
class CPUTrainingTests(unittest.TestCase):
    def backend(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.base=torch.nn.Parameter(torch.tensor([.1,-.1]),requires_grad=False)
                self.lora_A=torch.nn.Parameter(torch.tensor([[.5],[-.5]]));self.lora_B=torch.nn.Parameter(torch.zeros(1))
            def forward(self):return self.base+self.lora_A[:,0]*self.lora_B[0]
        class Backend:
            def __init__(self):self.model=Policy()
            def sample(self,question,seed):
                with torch.no_grad():y=int(torch.multinomial(self.model().softmax(0),1,generator=torch.Generator().manual_seed(seed)))
                return SimpleNamespace(context=(4,),tokens=(y,),terminated=True),'42' if y else '41'
            def log_prob(self,s):return self.model().log_softmax(0)[s.tokens[0]]
            def supervised(self,q,target):return SimpleNamespace(context=(4,),tokens=(1,),terminated=True)
        return Backend()

    def cases(self):return [cs.Case(str(i),'synthetic','t',str(i),'q'+str(i),'42','train','train','v1','numeric') for i in range(4)]

    def test_optimizer_only_updates_lora(self):
        b=self.backend();base=b.model.base.detach().clone();initial=b.model.lora_B.detach().clone();cases=self.cases()
        trainer=ct.ContentTrainer(b,cases,ct.TrainConfig(roots=4,k=4,max_tokens=1),cs.Scorer(cases))
        result=trainer.step()
        self.assertTrue(torch.equal(base,b.model.base));self.assertFalse(torch.equal(initial,b.model.lora_B));self.assertIsNone(b.model.base.grad)
        self.assertEqual(result['sampled_tokens'],16);self.assertEqual(result['beta'],0)

    def test_each_reference_runs_under_same_backend(self):
        for algorithm in ('rloo','grpo_reference','drgrpo_reference'):
            cases=self.cases();b=self.backend();engine=ct.ContentTrainer(b,cases,ct.TrainConfig(algorithm=algorithm,max_tokens=1),cs.Scorer(cases))
            self.assertTrue(math.isfinite(engine.step()['loss']))

    def test_sft_does_not_report_tautological_accuracy(self):
        b=self.backend();cases=self.cases();e=ct.ContentTrainer(b,cases,ct.TrainConfig(algorithm='sft_answers',max_tokens=1),cs.Scorer(cases))
        result=e.step();self.assertIsNone(result['train_accuracy']);self.assertEqual(result['sampled_tokens'],0);self.assertEqual(result['supervised_target_tokens'],4)

    def test_train_data_and_trainable_base_guards(self):
        cases=self.cases();cases[0]=replace(cases[0],split='dev')
        with self.assertRaises(ValueError):ct.ContentTrainer(self.backend(),cases,ct.TrainConfig(),cs.Scorer(cases))
        b=self.backend();b.model.base.requires_grad_(True)
        with self.assertRaises(ValueError):ct.ContentTrainer(b,self.cases(),ct.TrainConfig(),cs.Scorer(self.cases()))

    def test_whole_group_reservation(self):
        cases=self.cases();e=ct.ContentTrainer(self.backend(),cases,ct.TrainConfig(max_tokens=10,k=4,max_sampled_tokens=39),cs.Scorer(cases))
        with self.assertRaises(StopIteration):e.step()

    def test_rloo_exact_gradient_under_two_independent_draws(self):
        x=torch.tensor(.3,dtype=torch.float64,requires_grad=True);log=torch.stack((x*0,x)).log_softmax(0)
        exact=log[1].exp();surrogate=x*0
        for a in (0,1):
            for b in (0,1):
                weight=float((log[a]+log[b]).detach().exp());adv=ct.advantages([a,b],'rloo')
                surrogate+=weight*(adv[0]*log[a]+adv[1]*log[b])/2
        g1=torch.autograd.grad(exact,x,retain_graph=True)[0];g2=torch.autograd.grad(surrogate,x)[0]
        self.assertAlmostEqual(float(g1),float(g2),places=12)

    def test_degrpo_control_response_mask_and_gradient(self):
        old=torch.zeros((2,3));new=torch.zeros((2,3),requires_grad=True);a=torch.tensor([[1.,2.,2.],[3.,4.,4.]])
        loss=ct.degrpo_reference_loss(old,new,a,torch.ones_like(new),alpha=.1)
        self.assertAlmostEqual(float(loss.detach()),-3.2,places=6);loss.backward()
        self.assertAlmostEqual(float(new.grad[0,0]),-.05,places=6);self.assertAlmostEqual(float(new.grad[0,1]),-.5,places=6)

    def test_degrpo_missing_control_rejected(self):
        log=torch.zeros(1,2);mask=torch.tensor([[0.,1.]])
        with self.assertRaises(ValueError):ct.degrpo_reference_loss(log,log,log,mask)


@unittest.skipIf(craft_core is None,'full CRAFT checkout required; dedicated CI exercises integration')
class CraftIntegrationTests(unittest.TestCase):
    def test_transformed_utility_preserves_counterfactual_estimator(self):
        cc=craft_core;cfg=cc.Config(samples_per_arm=1,cost_scale=100)
        s=cc.Segment((1,),(2,),True)
        arms=((cc.Branch((s,),0.,1.,''),),(cc.Branch((s,),1.,1000.,'42'),))
        f=cc.Fork('k','t',s,(3,),(.5,.5),arms,1.,0.,0.)
        out=ct.quality_credit(f,cfg,.2)
        self.assertAlmostEqual(out.utility,.3);self.assertGreater(out.gate[1],0)
        self.assertAlmostEqual(out.expected_cost,500.5)

    def test_no_success_cost_during_quality_warmup(self):
        cc=craft_core;cfg=cc.Config(samples_per_arm=1,cost_scale=100);s=cc.Segment((1,),(2,),True)
        f=cc.Fork('k','t',s,(3,),(.5,.5),((cc.Branch((s,),1.,1.,'42'),),(cc.Branch((s,),1.,1000.,'42'),)),1.,0.,0.)
        out=ct.quality_credit(f,cfg,0.);self.assertEqual(out.gate,(0.,0.));self.assertEqual(out.utility,1.)


if __name__=='__main__':unittest.main()
