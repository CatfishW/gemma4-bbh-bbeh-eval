import copy
from dataclasses import replace
import itertools
import math
import random
import unittest

from rl_craft.core import Branch, Config, Example, Fork, Scheduler, Segment, credit
try:
    import torch
except ImportError:
    torch = None


def seg(token=0):
    return Segment((7,), (token,), True)


def fork(cfg=None, outcomes=((1,0),(0,1)), costs=((3,4),(5,6)), p=(.4,.6)):
    return Fork("k","task",seg(),(8,),p,
                tuple(tuple(Branch((seg(),),r,c,"a") for r,c in zip(rs,cs)) for rs,cs in zip(outcomes,costs)),
                1.,.13,.7)


class CoreTests(unittest.TestCase):
    def test_configuration_rejects_invalid_values(self):
        for kwargs in ({"samples_per_arm":0},{"exploration":0},{"temperature":0},
                       {"target_accuracy":2},{"learning_rate":float("nan")},{"rank":True},
                       {"estimator":"wrong"},{"estimator":"sampled"},{"suffix_baseline":"own-reward"},
                       {"cost_weight":-1},{"seed":-3},{"quality_dual":1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_counterfactual_credit_is_probability_weighted(self):
        cfg = Config(samples_per_arm=1,cost_weight=0)
        f = replace(fork(outcomes=((1,),(0,)),costs=((3,),(5,)),p=(.25,.75)),multiplier=0)
        c = credit(f,cfg)
        self.assertAlmostEqual(c.prefix,.25-.13)
        self.assertEqual(c.gate,(.1875,-.1875))
        self.assertEqual(c.suffixes,((.25,),(-.75,)))

    def test_all_correct_has_cost_sensitive_stopping_credit(self):
        cfg = Config(samples_per_arm=1,cost_weight=1,cost_scale=1)
        c = credit(fork(outcomes=((1,),(1,)),costs=((3,),(7,))),cfg)
        self.assertGreater(c.gate[0],0)
        self.assertLess(c.gate[1],0)

    def test_own_trajectory_is_excluded_from_its_baseline(self):
        cfg = Config(cost_weight=0)
        f = fork()
        c = credit(f,cfg)
        changed = replace(f,arms=((replace(f.arms[0][0],reward=0),f.arms[0][1]),f.arms[1]))
        d = credit(changed,cfg)
        self.assertAlmostEqual(c.suffixes[0][0]-d.suffixes[0][0],.4*1.7/2)

    def test_uniform_task_objective_not_uniform_examples(self):
        rows = [Example("a","A","q","x",0),Example("b","A","q","x",1),Example("c","B","q","x",0)]
        s = Scheduler(rows,Config())
        self.assertEqual(s.target,{"a":.25,"b":.25,"c":.5})

    def test_importance_is_bounded_and_proposal_is_positive(self):
        rows = [Example(str(i),"t","q","a",i) for i in range(5)]
        s = Scheduler(rows,Config(exploration=.2))
        s.rows["0"].update(leverage=100,cost=.001)
        q = s.proposal()
        self.assertAlmostEqual(sum(q.values()),1)
        self.assertTrue(all(0<s.target[k]/v <= 5.00000001 for k,v in q.items()))
        draws = s.draw(random.Random(4),100)
        self.assertTrue(all(abs(w-s.target[k]/q[k])<1e-10 for k,w in draws))

    def test_snapshot_is_unchanged_by_selection(self):
        s = Scheduler([Example("k","task","q","a",0)],Config())
        state = s.state_dict()
        s.draw(random.Random(1),10)
        self.assertEqual(state,s.state_dict())
        state["rows"]["k"]["return"] = 999
        self.assertNotEqual(state,s.state_dict())

    def test_dual_increases_below_floor_and_is_capped(self):
        cfg = Config(target_accuracy=.9,dual_lr=100,dual_max=2)
        s = Scheduler([Example("k","task","q","a",0)],cfg)
        f = fork(outcomes=((0,0),(0,0)))
        s.update([f],[credit(f,cfg)])
        self.assertEqual(s.dual["task"],2)
        f = fork(outcomes=((1,1),(1,1)))
        s.update([f],[credit(f,cfg)])
        self.assertEqual(s.dual["task"],0)

    def test_disabled_dual_and_custom_task_floor(self):
        cfg = Config(quality_dual=False)
        s = Scheduler([Example("k","task","q","a",0)],cfg,{"task":.1})
        s.update([fork()],[credit(fork(),cfg)])
        self.assertEqual(s.dual["task"],0)
        with self.assertRaises(ValueError):
            Scheduler([Example("k","task","q","a",0)],cfg,{"wrong":.4})

    def test_state_restore_rejects_changed_tasks_or_targets(self):
        s = Scheduler([Example("k","task","q","a",0)],Config())
        saved = s.state_dict()
        s.load_state_dict(saved)
        saved["targets"]["task"] = .99
        with self.assertRaises(ValueError):
            s.load_state_dict(saved)

    def test_invalid_tree_never_silently_normalizes_probabilities(self):
        for value in ((0,1),(.5,.6),(float("nan"),.5)):
            with self.assertRaises(ValueError):
                credit(replace(fork(),probabilities=value),Config())
        with self.assertRaises(ValueError):
            credit(fork(),Config(samples_per_arm=1))

    def test_generation_budget_counts_both_counterfactual_arms(self):
        cfg = Config(prefix_tokens=3,continue_tokens=5,answer_tokens=7,samples_per_arm=2)
        self.assertEqual(cfg.worst_tree_tokens,3+2*(5+14))
        sampled = replace(cfg,estimator="sampled",samples_per_arm=1)
        self.assertEqual(sampled.worst_tree_tokens,15)


@unittest.skipIf(torch is None,"optional torch not installed")
class ExactGradientTests(unittest.TestCase):
    """Enumerate every outcome; do not rely on a noisy Monte Carlo smoke claim."""
    def check_estimator(self, estimator, samples, baseline="crossfit"):
        theta = torch.tensor([.3,-.6,.8,.5,-.2,-.4,.9],dtype=torch.float64,requires_grad=True)
        cfg = Config(estimator=estimator,samples_per_arm=samples,suffix_baseline=baseline,
                     cost_weight=.23,cost_scale=2)
        root = torch.stack((theta[0]*0,theta[0])).log_softmax(0)
        gates = [torch.stack((theta[1+x]*0,theta[1+x])).log_softmax(0) for x in (0,1)]
        ys = [[torch.stack((theta[3+2*a+x]*0,theta[3+2*a+x])).log_softmax(0) for a in (0,1)] for x in (0,1)]
        utility = lambda x,a,y: 1.7*float((x^a^y)==1)-.23*(2+2*a+.3*y+.1*x)/2
        exact = sum(root[x].exp()*gates[x][a].exp()*ys[x][a][y].exp()*utility(x,a,y)
                    for x,a,y in itertools.product((0,1),repeat=3))
        expected_surrogate = theta.sum()*0
        for x in (0,1):
            if estimator == "paired":
                for leaves in itertools.product((0,1),repeat=2*samples):
                    arms = tuple(tuple(Branch((seg(y),),float((x^a^y)==1),2+2*a+.3*y+.1*x,str(y))
                                       for y in leaves[a*samples:(a+1)*samples]) for a in (0,1))
                    f = Fork("k","task",seg(x),(9,),tuple(float(p) for p in gates[x].detach().exp()),arms,1,.13,.7)
                    c = credit(f,cfg)
                    weight = float(root[x].detach().exp())
                    surrogate = c.prefix*root[x]+sum(c.gate[a]*gates[x][a] for a in (0,1))
                    for a in (0,1):
                        for j,y in enumerate(leaves[a*samples:(a+1)*samples]):
                            weight *= float(ys[x][a][y].detach().exp())
                            surrogate = surrogate+c.suffixes[a][j]*ys[x][a][y]
                    expected_surrogate = expected_surrogate+weight*surrogate
            else:
                for a,y in itertools.product((0,1),repeat=2):
                    b = Branch((seg(y),),float((x^a^y)==1),2+2*a+.3*y+.1*x,str(y))
                    arms = ((b,),()) if a==0 else ((),(b,))
                    f = Fork("k","task",seg(x),(9,),tuple(float(p) for p in gates[x].detach().exp()),arms,1,.13,.7,a)
                    c = credit(f,cfg)
                    weight = float((root[x]+gates[x][a]+ys[x][a][y]).detach().exp())
                    surrogate = c.prefix*root[x]+c.gate[a]*gates[x][a]+c.suffixes[a][0]*ys[x][a][y]
                    expected_surrogate = expected_surrogate+weight*surrogate
        true_gradient = torch.autograd.grad(exact,theta,retain_graph=True)[0]
        estimated_gradient = torch.autograd.grad(expected_surrogate,theta)[0]
        self.assertTrue(torch.allclose(true_gradient,estimated_gradient,atol=1e-12,rtol=1e-12),
                        f"gradient error={float((true_gradient-estimated_gradient).abs().max())}")
        self.assertGreater(float(true_gradient.abs().sum()),.01)

    def test_paired_single_sample_each_arm_is_unbiased(self):
        self.check_estimator("paired",1)

    def test_paired_two_siblings_each_arm_is_unbiased(self):
        self.check_estimator("paired",2)

    def test_historical_suffix_baseline_is_unbiased(self):
        self.check_estimator("paired",2,"history")

    def test_sampled_action_ablation_is_unbiased(self):
        self.check_estimator("sampled",1)

    def test_adaptive_proposal_recovers_target_gradient_exactly(self):
        parameter = torch.tensor(.2,dtype=torch.float64,requires_grad=True)
        target,q = [.25,.25,.5],[.8,.1,.1]
        utilities = [parameter.sin(),parameter**2,parameter.exp()]
        exact = sum(p*u for p,u in zip(target,utilities))
        corrected = sum(proposal*(p/proposal)*u for p,proposal,u in zip(target,q,utilities))
        g1 = torch.autograd.grad(exact,parameter,retain_graph=True)[0]
        g2 = torch.autograd.grad(corrected,parameter)[0]
        self.assertAlmostEqual(float(g1),float(g2),places=12)


if __name__ == "__main__":
    unittest.main()
