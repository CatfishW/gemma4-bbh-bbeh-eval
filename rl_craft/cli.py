"""Run with python -m rl_craft. Training is opt-in; smoke requires no downloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

from .core import Config, digest
from .data import (file_hash, load_rows, model_identity, prepare, source_identity,
                   training_data, write_json)


def config_from(path: Path | None) -> Config:
    return Config(**(json.loads(path.read_text()) if path else {}))


def behavior_signature(cfg: Config) -> dict:
    keys = ("prefix_tokens","continue_tokens","answer_tokens","temperature","gate_temperature","max_context")
    return {k:getattr(cfg,k) for k in keys}


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if not 0 <= wins <= n or n < 1:
        raise ValueError("invalid reference outcomes")
    p = wins/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def save_checkpoint(trainer, output: Path, protocol: dict, metrics: list[dict], traces: list[dict]) -> Path:
    import torch
    name = f"checkpoint-{trainer.iteration:05d}"
    target, temp = output/name, output/(name+".tmp")
    if target.exists() or temp.exists():
        raise FileExistsError("checkpoint path exists; refusing overwrite")
    temp.mkdir()
    try:
        trainer.backend.save_adapter(temp/"adapter")
        torch.save(trainer.state_dict(),temp/"state.pt")
        write_json(temp/"meta.json",{"config":asdict(trainer.cfg),"protocol_digest":digest(protocol),
                                    "model_identity":protocol["model_identity"],"iteration":trainer.iteration})
        write_json(temp/"metrics.json",metrics)
        write_json(temp/"traces.json",traces)
        hashes = {str(p.relative_to(temp)):file_hash(p) for p in sorted(temp.rglob("*")) if p.is_file()}
        write_json(temp/"hashes.json",hashes)
        os.replace(temp,target)  # atomic on the same filesystem
        pointer = output/"latest.tmp"
        pointer.write_text(name+"\n")
        os.replace(pointer,output/"latest")
    except BaseException:
        # Preserve failed temp files for inspection, never mistake them for a checkpoint.
        raise
    return target


def verify_checkpoint(path: Path) -> dict:
    hashes = json.loads((path/"hashes.json").read_text())
    if not hashes or any(Path(name).is_absolute() or ".." in Path(name).parts for name in hashes):
        raise ValueError("invalid checkpoint file manifest")
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p.name != "hashes.json"}
    if actual != set(hashes) or any(file_hash(path/name)!=h for name,h in hashes.items()):
        raise ValueError("checkpoint file integrity failure")
    return json.loads((path/"meta.json").read_text())


def train(args, toy: bool = False) -> dict:
    import torch
    from .trainer import Trainer
    if toy:
        from .toy import ToyBackend, examples
        cfg = Config(iterations=args.iterations, roots_per_step=4, samples_per_arm=2,
                     prefix_tokens=1,continue_tokens=1,answer_tokens=1,cost_scale=4,
                     learning_rate=0.03,rank=2,alpha=2,max_sampled_tokens=100000,seed=7)
        rows, data_manifest = examples(), {"source":"synthetic-generated", "split":"calibration"}
        identity = {"digest":"toy-deterministic-base-v1", "files":{}}
        model_path = None
        scorer = lambda prediction,target: prediction == target
    else:
        from eval_benchmarks import evaluate_correctness
        cfg = config_from(args.config)
        rows, data_manifest = training_data(args.data)
        model_path = str(args.model_path.resolve(strict=True))
        identity = model_identity(args.model_path)
        scorer = evaluate_correctness
    targets = None
    target_payload = None
    if not toy and args.targets:
        target_payload = json.loads(args.targets.read_text())
        if (target_payload.get("schema") != 1 or target_payload.get("model_digest") != identity["digest"]
            or target_payload.get("data_sha256") != data_manifest["sha256"]
            or target_payload.get("behavior") != behavior_signature(cfg)
            or target_payload.get("code") != source_identity()):
            raise ValueError("reference floors do not match data/model/code/behavior")
        targets = target_payload["targets"]
    versions = {}
    for name in ("torch","transformers","peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    protocol = {"schema":1,"method":"CRAFT","base_branch":"rl-volt@e0dab8a08af7ab4dc2b95ce790112820b733afab",
                "config":asdict(cfg),"data":data_manifest,"model_path":model_path,"model_identity":identity,
                "code":source_identity(),"versions":versions,"targets":target_payload,
                "task_targets_source":"frozen-base-calibration" if targets is not None else "explicit-fixed-config",
                "toy":toy,"scorer":"exact-string" if toy else "eval_benchmarks.evaluate_correctness",
                "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "study_id":getattr(args,"study_id","synthetic-smoke"),
                "claim":"exploratory implementation; not a confirmatory benchmark result"}
    output = args.output
    resumed = None
    if args.resume:
        if json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("resume protocol differs: data, code, versions, model, or config changed")
        name = (output/"latest").read_text().strip()
        if "/" in name or not name.startswith("checkpoint-") or "." in name:
            raise ValueError("invalid latest pointer")
        resumed = output/name
        meta = verify_checkpoint(resumed)
        if meta["protocol_digest"] != digest(protocol):
            raise ValueError("checkpoint/protocol mismatch")
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"protocol.json",protocol)  # before any stochastic model operation
    torch.manual_seed(cfg.seed)
    if toy:
        from .toy import ToyBackend
        backend = ToyBackend(cfg)
        if resumed:
            backend.load_adapter(resumed/"adapter")
    else:
        from .hf_backend import HFBackend
        backend = HFBackend(model_path,cfg,args.device,str(resumed/"adapter") if resumed else None)
    engine = Trainer(backend,rows,cfg,scorer,targets)
    metrics, traces = [], []
    if resumed:
        # Restricted deserialization; optimizer/RNG state contains tensors and primitives only.
        state = torch.load(resumed/"state.pt",map_location="cpu",weights_only=True)
        engine.load_state_dict(state)
        metrics = json.loads((resumed/"metrics.json").read_text())
        traces = json.loads((resumed/"traces.json").read_text())
    before_base = None
    if toy:
        before_base = backend.model.base.detach().clone()
    stop_reason = "iterations-complete"
    started = time.perf_counter()
    try:
        if resumed is None:
            save_checkpoint(engine,output,protocol,metrics,traces)
        while engine.iteration < cfg.iterations:
            try:
                metric, forks = engine.step()
            except StopIteration as exc:
                stop_reason = str(exc)
                break
            metrics.append(metric)
            # Integer token IDs include source-derived/model text. Never upload
            # these traces by default; keep with protected dataset artifacts.
            traces.append({"iteration":engine.iteration,"forks":[asdict(f) for f in forks]})
            save_checkpoint(engine,output,protocol,metrics,traces)
            print(json.dumps(metric,sort_keys=True),flush=True)
    except BaseException as exc:
        # The prior atomic checkpoint remains the only resumable state.
        failure = output/f"failure-{time.time_ns()}.json"
        write_json(failure,{"type":type(exc).__name__,"message":str(exc),"next_iteration":engine.iteration+1})
        raise
    report = {"iterations":engine.iteration,"sampled_tokens":engine.sampled_tokens,"stop_reason":stop_reason,
              "checkpoint":(output/"latest").read_text().strip(),"protocol_digest":digest(protocol),
              "this_invocation_seconds":time.perf_counter()-started,"real_model_run":not toy,
              "base_unchanged":bool(torch.equal(before_base,backend.model.base)) if toy else "base parameters frozen; no full post-run byte comparison",
              "final_metrics":metrics[-1] if metrics else None}
    report_path = output/f"summary-{time.time_ns()}.json"
    write_json(report_path,report)
    return report


def calibrate(args) -> dict:
    from eval_benchmarks import evaluate_correctness
    from .hf_backend import HFBackend
    from .trainer import predict
    cfg = config_from(args.config)
    rows, manifest = training_data(args.data)
    identity = model_identity(args.model_path)
    if not 0 <= args.tolerance <= 1:
        raise ValueError("invalid floor tolerance")
    args.output.mkdir(parents=True,exist_ok=False)
    header = {"schema":1,"model_digest":identity["digest"],"data_sha256":manifest["sha256"],
              "behavior":behavior_signature(cfg),"code":source_identity(),"tolerance":args.tolerance,
              "control":"initial untrained LoRA = base, always-continue", "split":"calibration"}
    write_json(args.output/"protocol.json",header)
    backend = HFBackend(str(args.model_path),cfg,args.device)
    by_task, records = {}, []
    for e in rows:
        result = predict(backend,e.question,cfg,seed=cfg.seed+100000,mode="always-continue")
        correct = result["answer_terminated"] and evaluate_correctness(result["prediction"],e.target)
        by_task.setdefault(e.task,[]).append(bool(correct))
        records.append({"key":e.key,"task":e.task,"correct":bool(correct),**result})
    payload = {**header,"targets":{t:max(0.0,wilson_lower(sum(rs),len(rs))-args.tolerance) for t,rs in by_task.items()},
               "counts":{t:{"n":len(rs),"correct":sum(rs)} for t,rs in by_task.items()},
               "bound_note":"Wilson descriptive calibration bound, not a finite-sample deployment guarantee"}
    write_json(args.output/"targets.json",payload)
    write_json(args.output/"predictions.json",records)
    return payload


def evaluate(args) -> dict:
    from .hf_backend import HFBackend
    from .trainer import predict
    from eval_benchmarks import evaluate_correctness
    meta = verify_checkpoint(args.checkpoint)
    cfg = Config(**meta["config"])
    if model_identity(args.model_path) != meta["model_identity"]:
        raise ValueError("base checkpoint bytes differ from training")
    rows = load_rows(args.data)
    if args.split == "test" and not args.allow_test:
        raise ValueError("test requires --allow-test and a newly declared --study-id")
    if any(not (25 <= e.index < 50 if args.split == "validation" else e.index >= 50 if args.split == "test" else True) for e in rows):
        raise ValueError("mixed/wrong split indices; do not silently skip rows")
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"protocol.json",{"checkpoint":str(args.checkpoint),"checkpoint_meta":meta,
                                          "data_sha256":file_hash(args.data),"split":args.split,"gate":args.gate,
                                          "study_id":args.study_id,"code":source_identity()})
    backend = HFBackend(str(args.model_path),cfg,args.device,str(args.checkpoint/"adapter"))
    records = []
    with (args.output/"predictions.jsonl").open("x") as stream:
        for e in rows:
            result = predict(backend,e.question,cfg,seed=cfg.seed,mode=args.gate)
            result.update(key=e.key,task=e.task,index=e.index,
                          correct=bool(result["answer_terminated"] and evaluate_correctness(result["prediction"],e.target)))
            stream.write(json.dumps(result)+"\n")
            stream.flush()
            records.append(result)
    by_task = {t:[r for r in records if r["task"]==t] for t in {e.task for e in rows}}
    times = sorted(r["elapsed_seconds"] for r in records)
    report = {"n":len(records),"micro_accuracy":sum(r["correct"] for r in records)/len(records),
              "macro_task_accuracy":sum(sum(r["correct"] for r in rs)/len(rs) for rs in by_task.values())/len(by_task),
              "per_task":{t:{"n":len(rs),"correct":sum(r["correct"] for r in rs)} for t,rs in by_task.items()},
              "mean_generated_tokens":sum(r["generated_tokens"] for r in records)/len(records),
              "mean_elapsed_seconds":sum(times)/len(times),"p95_seconds":times[math.ceil(.95*len(times))-1],
              "gate":args.gate,"stop_fraction":sum(r["action"]=="stop" for r in records)/len(records),
              "final_answer_truncations":sum(not r["answer_terminated"] for r in records),
              "parse_failures":backend.parse_failures,
              "mean_stop_probability":sum(r["probabilities"][0] for r in records)/len(records)}
    write_json(args.output/"summary.json",report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command",required=True)
    prep = commands.add_parser("prepare")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",type=Path)
    source.add_argument("--datasets-root",type=Path)
    prep.add_argument("--output",type=Path,required=True)
    prep.add_argument("--holdout-stride",type=int,default=4)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output",type=Path,required=True)
    smoke.add_argument("--iterations",type=int,default=12)
    smoke.add_argument("--resume",action="store_true")
    for command in ("train","calibrate-targets"):
        q = commands.add_parser(command)
        q.add_argument("--data",type=Path,required=True)
        q.add_argument("--model-path",type=Path,required=True)
        q.add_argument("--config",type=Path)
        q.add_argument("--device",default="cuda:0")
        q.add_argument("--output",type=Path,required=True)
        if command == "train":
            q.add_argument("--targets",type=Path)
            q.add_argument("--study-id",required=True)
            q.add_argument("--resume",action="store_true")
        else:
            q.add_argument("--tolerance",type=float,default=.02)
    q = commands.add_parser("evaluate")
    q.add_argument("--checkpoint",type=Path,required=True)
    q.add_argument("--model-path",type=Path,required=True)
    q.add_argument("--data",type=Path,required=True)
    q.add_argument("--device",default="cuda:0")
    q.add_argument("--split",choices=["validation","test","external"],required=True)
    q.add_argument("--study-id",required=True)
    q.add_argument("--allow-test",action="store_true")
    q.add_argument("--gate",choices=["sample","greedy","always-stop","always-continue"],default="sample")
    q.add_argument("--output",type=Path,required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare(args.input,args.datasets_root,args.output,args.holdout_stride)
        elif args.command == "smoke":
            report = train(args,toy=True)
        elif args.command == "train":
            report = train(args)
        elif args.command == "calibrate-targets":
            report = calibrate(args)
        else:
            report = evaluate(args)
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0
    except (ValueError,TypeError,OSError,ImportError) as exc:
        p.exit(2,f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
