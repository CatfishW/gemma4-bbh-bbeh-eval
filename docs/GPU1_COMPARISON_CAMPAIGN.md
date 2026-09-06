# GPU 1 full comparison campaign

The recovered 34-file comparison patch is integrated with the GPU-tested CRAFT
branch. The executable campaign is `comparison_campaign.py`; the previous
12-step/68-case BF16 pilot remains a separate experiment. This campaign uses
physical GPU 1 only and never stops other jobs or services.

## Frozen scope

| Component | Scope |
|---|---|
| Training | CRAFT-Q, reference RLOO, reference GRPO, reference Dr.GRPO, answer-only SFT |
| Training seeds | 7, 17, 27 |
| Schedule | Original 200 updates or 1,000,000 action-token ceiling; roots=4, k=4, rank=32, alpha=64, learning rate=2e-5 |
| Training data | 13,474 problems from official GSM8K/MATH train |
| Development data | 1,497 whole-question groups, seed 7, excluded from training |
| Math test | Complete GSM8K test (1,319) and MATH-500 (500) |
| Transfer test | Complete frozen BBH/BBEH/USR test split (9,550) |
| Prompt controls, math | Direct, CoT, CoD, plan-and-solve, self-verification, self-consistency k=3, self-ranking k=3, self-refinement, native thinking |
| Prompt controls, transfer | Direct, CoT, CoD, self-consistency k=3 |
| Trained controls, both cohorts | All 15 adapters, without test-based checkpoint selection |
| Additional released model | Thinkless-1.5B-RL-DeepScaleR, common CoT harness on full math test |
| Inference seed | 7 for every arm, shared across independent training seeds |

This is 15 training jobs, 44 full-cohort evaluations, and two paired comparison
jobs. All jobs are declared before any campaign result is used. The nine prompt
strategies retain their explicit registered token ceilings; they are not claimed
to have equal realized cost. The training schedules are complete configured runs,
not full epochs over every training problem. CRAFT samples more outcomes per
prefix than reference controls. Report both the step limit and actual token cost.

## Data provenance and defects

Pinned revisions are in `comparison_prepare_campaign.py` and
`comparison_download_data.py`:

- [Official GSM8K repository](https://github.com/openai/grade-school-math), commit
  `3101c7d5072418e28b9008a6636bde82a006892c`.
- [MATH export](https://huggingface.co/datasets/EleutherAI/hendrycks_math), revision
  `21a5633873b6a120296cce3e2df9d5550074f4a3`.
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

The MATH source contains 7,500 training records. Two have empty boxed reference
answers (`number_theory/661`, `number_theory/663`). Their exact source hashes are
recorded; they are excluded before splitting without inventing labels. Two other
records use valid unbraced single-atom TeX boxes, which the converter now accepts.
No evaluation rows are excluded. JSONL parsing preserves Unicode separators
inside strings. Split-local row indices are namespaced, and question/group/ID
leakage is checked before writing the manifests. This is an exact-match audit,
not a claim of semantic decontamination or absence from pretraining.

## Runtime changes

Training uses NF4 double quantization with BF16 computation, LoRA, CPU activation
offload, and an 8 GiB PyTorch allocator cap. The activation-offload path selects
the math SDPA kernel for backward compatibility with restored attention-mask
strides. Frozen default BF16 behavior remains available.

The largest test question contains 96,238 tokens before the stage prompt.
Evaluation permits the model's 131,072-token context, offloads KV caches, and
prefills in 64-token chunks. A process-local SDPA wrapper avoids a materialized
GQA KV repeat for long single-token decoding with Gemma's 512-dimensional heads.
It uses a zero-stride view and the same attention equation; it does not alter the
installed Transformers files or other processes. Numerical differences from
chunked BF16 execution are expected and checked at the next-token distribution.

Checkpoint directories atomically save adapter, optimizer, sampler state, Torch
RNG, CUDA RNG, scheduler, metrics, and integrity manifests. Resume requires the
same model/data/code/environment/configuration. Evaluation resumes a complete
JSONL case prefix under the identical protocol. Completed artifacts are checked
against their protocol and prediction hashes. A resource failure stops the
worker with logs; it is not converted into a wrong model answer.

The isolated venv reuses the existing Gemma runtime via a `.pth` entry and adds
pinned math/data dependencies. See `envs/comparison-gpu1.txt`. Reusing this runtime
assumes its installed versions stay fixed; the run records their identities.

## Launch and inspect

On `benwulab-remote`, from the immutable repository snapshot:

```bash
CUDA_VISIBLE_DEVICES= python comparison_download_data.py --root /path/to/run --thinkless
CUDA_VISIBLE_DEVICES= python comparison_prepare_campaign.py \
  --sources /path/to/run/data-sources --output /path/to/run/frozen-data-v1 \
  --legacy-root /data/benwulab/gemma4-eval/datasets
CUDA_VISIBLE_DEVICES= python comparison_campaign.py build \
  --root /path/to/run --model /data/models/gemma-4-E2B-it
nohup python -u comparison_campaign.py execute \
  --plan /path/to/run/campaign-v1/plan.json > /path/to/run/worker.log 2>&1 &
```

The worker sets `CUDA_VISIBLE_DEVICES=1` for each GPU child, runs one at a time,
waits for 9 GiB free before starting, and holds a single-worker file lock. All
inputs and source hashes are frozen. It updates `status.json`, per-job logs,
`REPORT.md`, and `aggregate.json`. A failed job stops the queue; after resolving
its cause, invoke the same execute command to use verified resume paths. Code
changes require a new campaign identity rather than silently resuming a changed
protocol. A long-running queue is not a completed experiment.

## Interpretation

The reference trainers implement fresh-policy single updates; they are not full
upstream PPO/DeGRPO pipelines. [Thinkless](https://github.com/VainF/Thinkless)
already covers short/long reasoning control and separate controller/content
losses. Its released checkpoint revision
`73551801776c64ecd599c46229120daae4934aec` is evaluated under this common harness,
not relabeled as a same-base or author-recipe reproduction. CRAFT's proposed
contribution is the same-prefix counterfactual estimator and credit allocation.

Paired comparisons retain group and task bootstrap intervals, McNemar tests,
Holm corrections, token costs, truncations, and errors. Training-seed uncertainty
must be reported across the three adapters separately from within-test bootstrap
uncertainty. Shared services make latency observational; the campaign deliberately
marks load profiles as uncontrolled. A SOTA claim needs completed evidence and
stronger author-recipe comparisons, not these implementation tests or a partial run.

## Scorer audit and campaign v2

The first launch was superseded before evaluation after an additional gold
round-trip audit. Version 2 applies common final-answer extraction before BBEH
scoring, and parses the final MATH prediction with the same math delimiters as
its reference. This fixes 109/500 MATH-500 bare symbolic gold answers that the
original asymmetric parsing could reject. A later final-answer marker also
supersedes an earlier boxed intermediate. Tests include incorrect symbolic
alternatives, not just matching gold strings.

All 500 MATH-500 gold answers now pass the round-trip check. Eight Linguini
references fail their own pinned upstream BBEH round-trip because its prediction
normalizer removes terminal periods while its reference normalizer retains them.
The primary scorer retains the upstream rule and all eight cases; this limitation
is recorded in the audit rather than silently rewriting test references.

The active directory is `campaign-v2`, built using `--name campaign-v2`. The
superseded v1 artifacts remain on the host with an explicit exclusion reason and
are not used for model comparison. The patch's first CI run passed; the scorer
correction adds real Math-Verify regression coverage to CI.
