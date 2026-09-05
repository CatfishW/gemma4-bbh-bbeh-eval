import copy
from collections import UserDict
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl_craft.core import Config, Example, digest
from rl_craft.data import load_rows, prepare, training_data, write_json
try:
    import torch
except ImportError:
    torch = None


class DataTests(unittest.TestCase):
    def rows(self):
        return [Example(f"suite/task{t}/{i}",f"suite/task{t}",f"question {i}","a",i) for t in range(5) for i in range(2)]

    def write_rows(self,path,rows):
        path.write_text("".join(json.dumps(asdict(r))+"\n" for r in rows))

    def test_prepare_excludes_tasks_without_accessing_test(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); src=root/"source.jsonl";out=root/"train.jsonl"
            self.write_rows(src,self.rows())
            m=prepare(src,None,out,4)
            rows,_=training_data(out)
            self.assertEqual(m["heldout_tasks"],["suite/task0","suite/task4"])
            self.assertEqual(len(rows),6)
            self.assertTrue(all(e.index<25 for e in rows))

    def test_test_contamination_duplicate_and_changed_file_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/"src";out=root/"out"
            self.write_rows(src,self.rows()+[replace(self.rows()[0],key="test",index=50)])
            with self.assertRaises(ValueError): prepare(src,None,out,0)
            self.write_rows(src,self.rows()+[self.rows()[0]])
            with self.assertRaises(ValueError): load_rows(src)
            self.write_rows(src,self.rows());prepare(src,None,out,0)
            out.write_text(out.read_text()+"\n")
            with self.assertRaises(ValueError): training_data(out)

    def test_no_silent_overwrite_or_empty_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/"src";out=root/"out"
            self.write_rows(src,[self.rows()[0]])
            with self.assertRaises(ValueError): prepare(src,None,out,4)
            prepare(src,None,out,0)
            with self.assertRaises(FileExistsError): prepare(src,None,out,0)


@unittest.skipIf(torch is None,"optional torch not installed")
class TrainingTests(unittest.TestCase):
    def config(self,**kw):
        values=dict(iterations=4,roots_per_step=4,samples_per_arm=2,prefix_tokens=1,continue_tokens=1,
                    answer_tokens=1,learning_rate=.03,cost_scale=4,rank=2,alpha=2,seed=7)
        values.update(kw)
        return Config(**values)

    def engine(self,cfg=None):
        from rl_craft.toy import ToyBackend,examples
        from rl_craft.trainer import Trainer
        cfg=cfg or self.config()
        return Trainer(ToyBackend(cfg),examples(),cfg,lambda a,b:a==b)

    def test_real_optimizer_updates_only_lora(self):
        engine=self.engine()
        before={n:p.detach().clone() for n,p in engine.backend.model.named_parameters()}
        metric,forks=engine.step()
        self.assertEqual(metric["iteration"],1)
        self.assertEqual(metric["sampled_tokens"],28)
        self.assertTrue(torch.equal(before["base"],engine.backend.model.base))
        self.assertFalse(torch.equal(before["lora_B"],engine.backend.model.lora_B))
        self.assertIsNone(engine.backend.model.base.grad)
        self.assertTrue(all(torch.isfinite(p).all() for p in engine.parameters))
        self.assertEqual(len(forks),4)

    def test_sampling_and_gradient_use_identical_temperature(self):
        from rl_craft.core import Segment
        backend=self.engine().backend
        segment=Segment((2,),(1,),True)
        expected=(backend.model(2)/.37).log_softmax(-1)[1]
        self.assertTrue(torch.equal(backend.log_prob(segment,.37),expected))

    def test_resume_replays_next_update_bit_exactly(self):
        a=self.engine();a.step()
        state=copy.deepcopy(a.state_dict())
        weights=copy.deepcopy(a.backend.model.state_dict())
        _,forks_a=a.step()
        b=self.engine();b.backend.model.load_state_dict(weights);b.load_state_dict(state)
        _,forks_b=b.step()
        self.assertEqual([asdict(f) for f in forks_a],[asdict(f) for f in forks_b])
        self.assertTrue(all(torch.equal(p,b.backend.model.state_dict()[k]) for k,p in a.backend.model.state_dict().items()))
        self.assertEqual(a.scheduler.state_dict(),b.scheduler.state_dict())

    def test_invalid_resume_config_or_trainable_base_fails(self):
        from rl_craft.trainer import lora_parameters
        a=self.engine();saved=a.state_dict();saved["config"]["temperature"]=.99
        with self.assertRaises(ValueError): a.load_state_dict(saved)
        a.backend.model.base.requires_grad_(True)
        with self.assertRaises(ValueError): lora_parameters(a.backend.model)

    def test_hard_budget_covers_all_forks(self):
        cfg=self.config(max_sampled_tokens=7)
        a=self.engine(cfg);metric,_=a.step()
        self.assertEqual(metric["roots"],1)
        self.assertEqual(a.sampled_tokens,7)
        with self.assertRaises(StopIteration): a.step()

    def test_dual_snapshot_is_not_changed_midbatch(self):
        a=self.engine(self.config(target_accuracy=1,dual_lr=1))
        _,forks=a.step()
        self.assertTrue(all(f.multiplier==0 for f in forks))
        self.assertGreater(sum(a.scheduler.dual.values()),0)

    def test_sampled_action_control_updates(self):
        a=self.engine(self.config(estimator="sampled",samples_per_arm=1,adaptive_allocation=False))
        metric,forks=a.step()
        self.assertTrue(all(f.selected_action in (0,1) for f in forks))
        self.assertTrue(all(sum(map(len,f.arms))==1 for f in forks))
        self.assertLessEqual(metric["sampled_tokens"],12)

    def test_no_validation_rows_can_train(self):
        from rl_craft.trainer import Trainer
        from rl_craft.toy import ToyBackend,examples
        with self.assertRaises(ValueError):
            Trainer(ToyBackend(),[replace(examples()[0],index=25)],self.config(),lambda a,b:a==b)

    def test_inference_receives_no_reward_and_does_not_update(self):
        from rl_craft.trainer import predict
        a=self.engine()
        before=copy.deepcopy(a.backend.model.state_dict())
        stopped=predict(a.backend,"Return bit 1",a.cfg,mode="always-stop")
        continued=predict(a.backend,"Return bit 1",a.cfg,mode="always-continue")
        self.assertEqual(stopped["generated_tokens"],2)
        self.assertEqual(continued["generated_tokens"],3)
        self.assertTrue(all(torch.equal(v,a.backend.model.state_dict()[k]) for k,v in before.items()))

    def test_one_prefix_per_family_not_per_leaf(self):
        a=self.engine()
        with patch.object(a.backend,"sample",wraps=a.backend.sample) as wrapped:
            a.step()
        # 4 roots * (1 prefix + 2 stop answers + 2*(continue + answer))
        self.assertEqual(wrapped.call_count,28)

    def test_failed_generation_does_not_update_weights_or_scheduler(self):
        a=self.engine();before=copy.deepcopy(a.backend.model.state_dict());snap=a.scheduler.state_dict()
        with patch.object(a.backend,"sample",side_effect=RuntimeError("OOM")):
            with self.assertRaises(RuntimeError): a.step()
        self.assertEqual(a.iteration,0)
        self.assertEqual(a.scheduler.state_dict(),snap)
        self.assertTrue(all(torch.equal(v,a.backend.model.state_dict()[k]) for k,v in before.items()))

    def test_truncated_final_gets_zero_reward(self):
        a=self.engine();original=a.backend.sample
        def truncated(*args,**kw):
            return replace(original(*args,**kw),terminated=False)
        with patch.object(a.backend,"sample",side_effect=truncated):
            metric,forks=a.step()
        self.assertTrue(all(b.reward==0 for f in forks for arm in f.arms for b in arm))
        self.assertEqual(metric["final_answer_truncations"],16)

    def test_smoke_cli_integrity_resume_and_nonoverwrite(self):
        from rl_craft.cli import main,verify_checkpoint
        with tempfile.TemporaryDirectory() as d,patch("sys.stdout",new_callable=io.StringIO):
            out=Path(d)/"run"
            args=["smoke","--output",str(out),"--iterations","2"]
            self.assertEqual(main(args),0)
            latest=out/(out/"latest").read_text().strip()
            self.assertEqual(verify_checkpoint(latest)["iteration"],2)
            self.assertEqual(main(args+["--resume"]),0)
            with patch("sys.stderr",new_callable=io.StringIO),self.assertRaises(SystemExit): main(args)
            (latest/"metrics.json").write_text("[]")
            with self.assertRaises(ValueError): verify_checkpoint(latest)

    def test_hf_chunked_logprob_matches_full_logits_and_gradients(self):
        from rl_craft.hf_backend import HFBackend
        from rl_craft.core import Segment
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(9,5)
            def forward(self,input_ids,attention_mask,use_cache):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        b=HFBackend.__new__(HFBackend);b.device=torch.device("cpu");b.config=self.config(max_context=128)
        b.backbone=Backbone();b.head=torch.nn.Linear(5,9);b.softcap=1.3
        segment=Segment((1,2,3),tuple(i%9 for i in range(65)),True)
        actual=b.log_prob(segment,.73)
        full=segment.context+segment.tokens
        hidden=b.backbone.embedding(torch.tensor(full))
        logits=b._head(hidden)[len(segment.context)-1:-1]/.73
        expected=logits.log_softmax(-1).gather(1,torch.tensor(segment.tokens)[:,None]).sum()
        params=list(b.backbone.parameters())+list(b.head.parameters())
        ga=torch.autograd.grad(actual,params,retain_graph=True);ge=torch.autograd.grad(expected,params)
        self.assertAlmostEqual(float(actual.detach()),float(expected.detach()),places=4)
        self.assertTrue(all(torch.allclose(x,y,atol=2e-5,rtol=2e-5) for x,y in zip(ga,ge)))

    def test_hf_prompt_accepts_batch_encoding_mapping(self):
        from rl_craft.hf_backend import HFBackend
        b=HFBackend.__new__(HFBackend);b.config=self.config()
        b.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:UserDict(input_ids=[2,3,4]))
        self.assertEqual(b.prompt("notes","Question"),(2,3,4))


if __name__ == "__main__":
    unittest.main()
