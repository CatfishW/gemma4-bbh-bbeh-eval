"""Explicit calibration manifests and local checkpoint identities."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from .core import Example, digest


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def load_rows(path: Path) -> list[Example]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = Example(**json.loads(line))
            if any(not isinstance(getattr(row,k),str) or not getattr(row,k) for k in ("key","task","question","target")):
                raise ValueError("all example strings must be nonempty")
            if type(row.index) is not int or row.index < 0:
                raise ValueError("invalid example index")
            rows.append(row)
    if not rows or len({e.key for e in rows}) != len(rows):
        raise ValueError("empty data or duplicate keys")
    return rows


def training_data(path: Path) -> tuple[list[Example], dict]:
    manifest = json.loads(Path(str(path)+".manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("split") != "calibration" or manifest.get("sha256") != file_hash(path):
        raise ValueError("missing/changed calibration manifest; run prepare first")
    rows = load_rows(path)
    if manifest.get("rows") != len(rows) or any(e.index >= 25 or e.task in manifest["heldout_tasks"] for e in rows):
        raise ValueError("heldout/test contamination or mismatched row count")
    return rows, manifest


def prepare(input_path: Path | None, datasets_root: Path | None, output: Path, stride: int = 4) -> dict:
    if stride < 0 or stride == 1:
        raise ValueError("holdout stride must be 0 (explicitly disabled) or >=2")
    revisions = {}
    if datasets_root is not None:
        from eval_benchmarks import load_bbh, load_bbeh, load_unpuzzles_simple_reasoning
        raw = [*load_bbh(datasets_root,25), *load_bbeh(datasets_root,25), *load_unpuzzles_simple_reasoning(datasets_root,25)]
        rows = [Example(f"{e.benchmark}/{e.task}/{e.index}", f"{e.benchmark}/{e.task}", e.input,e.target,e.index) for e in raw]
        for name in ("BIG-Bench-Hard", "bbeh", "unpuzzles_and_simple_reasoning"):
            path = datasets_root/name
            revision = subprocess.run(["git","-C",str(path),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
            dirty = subprocess.run(["git","-C",str(path),"status","--porcelain","--untracked-files=no"],capture_output=True,text=True,check=True).stdout.strip()
            if dirty:
                raise ValueError(f"tracked dataset changes: {name}")
            revisions[name] = revision
    else:
        rows = load_rows(input_path)
    if not rows or len({e.key for e in rows}) != len(rows) or any(e.index >= 25 for e in rows):
        raise ValueError("prepare accepts unique calibration examples only; never silently filter a test file")
    held = set()
    tasks = sorted({e.task for e in rows})
    benchmarks = {t.split("/",1)[0] for t in tasks}
    if stride:
        for benchmark in sorted(benchmarks):
            group = [t for t in tasks if t.split("/",1)[0] == benchmark]
            held.update(t for i,t in enumerate(group) if i%stride == 0)
    kept = [e for e in rows if e.task not in held]
    if not kept:
        raise ValueError("no training tasks after holdout; use more tasks or explicitly set --holdout-stride 0")
    manifest_path = Path(str(output)+".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("prepared files already exist")
    with output.open("x",encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(asdict(e),ensure_ascii=False)+"\n")
    result = {"schema":1,"split":"calibration","rows":len(kept),"heldout_tasks":sorted(held),
              "holdout_stride":stride,"sha256":file_hash(output),"dataset_revisions":revisions,
              "source":str(datasets_root or input_path),"source_rows":len(rows)}
    write_json(manifest_path,result)
    return result


def model_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    files = sorted(p for p in path.rglob("*") if p.is_file() and
                   (p.suffix in {".safetensors", ".bin", ".json", ".model", ".tiktoken"}))
    if not (path/"config.json").is_file() or not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("local model config and weight files required")
    hashes = {str(p.relative_to(path)):file_hash(p) for p in files}
    return {"files":hashes,"digest":digest(hashes)}


def source_identity() -> dict:
    folder = Path(__file__).resolve().parent
    paths = sorted(folder.glob("*.py"))
    paths += [p for p in (folder.parent/"rl/modeling.py",folder.parent/"eval_benchmarks.py") if p.exists()]
    return {str(p.relative_to(folder.parent)):file_hash(p) for p in paths}
