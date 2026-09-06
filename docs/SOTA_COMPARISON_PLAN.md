# Comparison programme: strong LLM augmentation baselines

**Status (2026-09-05): implementation and revised research plan, not a new SOTA result.**
This change is based on `rl-craft-20260904@ba2977c79d11229f63c08cbfa1cffcb80d17d2b8`,
including the user's completed GPU pilot and tokenizer/API fixes. Original CRAFT,
VOLT, GRPO, frozen-test artifacts and serving services are not rewritten.

The literature work here inspected author/upstream **GitHub artifacts**. General
live web/proceedings search was unavailable. The machine-readable
[`papers.json`](../experiments/comparison/papers.json) distinguishes inspected
primary artifacts from historical background and records immutable README blob
identities. This is a broad comparison set, not a certified exhaustive inventory
of the best methods available in September 2026. A paper's historical SOTA claim
is not evidence it is the present leader, nor that its number is comparable to a
new model/data/compute setting.

## 1. What the current evidence changes

The [completed GPU pilot](CRAFT_GPU_RESULTS.md) is real, but does not establish a
reasoning-quality gain: CRAFT and the untrained sampled gate both scored 15/68
(22.06%), with CRAFT using 75.3 rather than 96.0 generated tokens. The untrained
always-continue control scored 17/68; trained CRAFT always-continue scored 14/68.
All six conditions scored 3/20 on the five held-out task families. Seven of nine
CRAFT answer truncations were on USR logic. The pilot had 12 updates and visited
46 distinct training questions, not a full-scale reasoning experiment.

These results motivate **content quality first, compute reduction second**. The
next study must not optimize stop rate until the learned content policy is at
least competitive. Equality on 68 examples is not accuracy non-inferiority.

### The closest collision: Thinkless

[Thinkless](https://github.com/VainF/Thinkless) already learns whether to use short
or long reasoning and separates control-token and response losses with DeGRPO.
Its released recipe includes supervised warmup, a DeepScaleR RL dataset and a
1.5B reasoning checkpoint. It is therefore not defensible to claim that adaptive
thinking, or simply separating controller/content loss, is new in CRAFT.

The audited loss at
[`55817cb/.../core_algos.py`](https://github.com/VainF/Thinkless/blob/55817cbeedaf4fd862844cc3b471dbaf3aa43227/verl/verl/trainer/ppo/core_algos.py)
uses a first-token control mask, a remaining-response mask and a weighted sum of
the two clipped policy losses. `comparison_training.degrpo_reference_loss`
implements that mathematical split as a testable reference. **It is not the full
Thinkless training pipeline**, control-token setup, warmup or reproduction.

CRAFT's narrower research hypothesis is **paired stop/continue outcomes at the
same sampled prefix, correct shared-prefix/gate/suffix credit, and predictable
importance-corrected allocation**. Compare this against ordinary sampled-action
RL and the full upstream Thinkless recipe. Claim incremental mechanism gains
only if matched-compute ablations demonstrate them.

## 2. Mandatory comparison matrix

| Method / source | Track and relevance | Required experiment and caveat |
|---|---|---|
| [Chain of Thought](https://arxiv.org/abs/2201.11903), NeurIPS 2022; [Self-Consistency](https://arxiv.org/abs/2203.11171), ICLR 2023 | Frozen prompting; foundational quality/compute controls | Sweep total budgets; count all samples, not just the winning answer. Historical venues are background knowledge, not newly checked proceedings. |
| [Chain of Draft](https://github.com/sileix/chain-of-draft) | Frozen prompt compression; directly targets speed and quality | Exact five-word-step instruction is included, but our zero-shot single-user transfer differs from the upstream system-role few-shot recipe. The reported “as little as 7.6% of tokens” is not a universal guarantee. |
| [Self-Refine](https://github.com/madaan/self-refine) | Same-model feedback/revision | Include generation, feedback and revision costs. No test-label feedback or correctness-triggered retries. |
| [PAL](https://github.com/reasoning-machines/pal) | Program generation + interpreter | Essential baseline before claiming partial compilation is useful. A calculator-only shortcut is not a faithful PAL reproduction. |
| [Program of Thoughts](https://github.com/TIGER-AI-Lab/Program-of-Thoughts), **TMLR 2023** | Numerical reasoning with execution | Match interpreter, task, demonstrations and grader. Its README reports historical GSM8K artifacts with 1,318 examples: enforce equal actual cohorts rather than copy headline scores. |
| [Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm) | Task-specific search/evaluation | Use original Game24/crossword environments and charge value/vote calls. The repository distinguishes a 69% reproduced Game24 log from the original 74% run; preserve this distinction. |
| [Buffer of Thoughts](https://github.com/YangLing0818/buffer-of-thought-llm), **NeurIPS 2024 Spotlight** | Reusable thought-template modules | Closest memory comparator. Charge template construction, embeddings, retrieval and updates. Released three-task demo and later LightRAG math pipeline are different artifacts. |
| [SuperCorrect](https://github.com/YangLing0818/SuperCorrect-llm), **ICLR 2025** | Small-model template distillation/self-correction | Match teacher availability and training data, and avoid comparing its online judge with our exact/symbolic grader. |
| [ReasonFlux-PRM](https://github.com/Gen-Verse/ReasonFlux), **NeurIPS 2025** | Trajectory-aware learned reward module | Count the extra PRM model, candidate scoring, teacher/data-generation cost and total deployment memory. Distinguish ReasonFlux-Zero/F1 from PRM variants. |
| [s1](https://github.com/simplescaling/s1) | Supervised training plus budget forcing | Preserve tokenizer-specific thinking/final delimiters, s1 versus s1.1 teachers and datasets. A generic “Wait” suffix is not s1. |
| [Thinkless](https://github.com/VainF/Thinkless) | Learned short/long reasoning; closest training overlap | Match warmup, control tokens and RL corpus. Report upstream reproduction separately from same-base LoRA transfer. Conference status was not verified from the inspected README. |
| [ThinkPrune](https://github.com/UCSB-NLP-Chang/ThinkPrune) | Iterative RL reasoning-length pruning | Replicate actual cap schedule and initialization, not just a length penalty. Upstream environment is isolated. Venue not verified in this audit. |
| [Dr.GRPO](https://github.com/sail-sg/understand-r1-zero) | Normalization and template controls | Compare full upstream and a clearly labeled matched-runtime reference. Its README's ICML AI4Math workshop award is not a main-conference publication claim. |
| [RLOO](https://arxiv.org/abs/2402.14740), [GRPO](https://arxiv.org/abs/2402.03300) | Outcome-RL controls | Same LoRA, model, prompts, data and actual token/GPU budgets; one-update references here do not reproduce every upstream PPO/KL detail. |

**Implementation status is explicit.** Nine prompting configurations, controlled
content-RL trainers, CRAFT-Q and the common evaluation bridge are runnable here.
PAL, PoT, ToT, BoT, Thinkless, ThinkPrune, s1, SuperCorrect and ReasonFlux full
pipelines remain **upstream executions imported into the common scorer**, not
secretly replaced with superficial approximations. ReAct, Reflexion, Least-to-Most
and DAPO are secondary candidates recorded in the registry, not implemented or
claimed as freshly audited. This prevents a long baseline list from becoming a
false claim of a completed leaderboard.

## 3. Three tracks, two kinds of comparison

**Frozen prompting:** identical checkpoint, tokenizer, precision and serving
runtime; no hidden teacher or external executor. Direct, CoT, CoD, self-consistency,
self-ranking, bounded Self-Refine, planning and native thinking belong here.

**Module augmentation:** retain the same generator but declare each executor,
retriever, embedding model, memory, verifier and its costs. Full compilation is a
mandatory comparator for answer-stable partial compilation. Game24 is evaluated
with a real task verifier (all input numbers exactly once), not “answer equals 24.”

**Weight-trained augmentation:** separate SFT, RL and teacher-derived data. Count
base/generator parameters and extra modules separately. Keep CRAFT-Q versus RLOO
on the same training data and adapter rank; keep SuperCorrect/Thinkless on their
native recipes as distinct external-reproduction tracks.

For each method, distinguish **author-recipe reproduction** from **same-base
transfer**. A Gemma implementation of an idea may answer an important question
without reproducing a paper's absolute score. Comparing a 32B teacher-distilled
model with E2B is descriptive, not evidence that the method itself is better.

## 4. Data expansion and scale

[`datasets.json`](../experiments/comparison/datasets.json) lists 14 dataset groups
and their implementation/support boundaries.

The primary learning sources are **official GSM8K training data** and a declared
**MATH training subset**, with group-disjoint development data. Expand to
DeepScaleR or NuminaMath only after license/provenance and contamination audits.
The old benchmark-prefix training set is too narrow to establish broad gains;
do not fake arbitrary external examples as indices 0–24 to satisfy the old trainer.
The new trainers explicitly accept official `source_split=train` records instead.

Evaluation spans full BBH/BBEH/USR transfer, GSM8K, MATH-500, SVAMP or GSM-hard,
MMLU-Pro and GPQA-Diamond. AIME 2024/2025 are difficult small-N stress tests, not
the primary significance claim. Game24 supplies a direct ToT/BoT/PAL overlap
setting using upstream task verification. Do not pool MATH-500 with full MATH
test as independent observations; do not pool perturbed puzzles with their source
questions as independent evidence. Repeated decoding trials do not create new
independent questions.

Source split and experiment role are separate fields. A public mirror called
`train` does not authorize training on a historically evaluation-only dataset.
The legacy exporter preserves original/shifted/unpuzzle groups from their source
puzzle identity, not task-local indices. Exact duplicate and declared group
leakage checks are implemented. **Semantic/near-duplicate decontamination is not
automated by a string hash** and remains a required corpus audit.

The suite does not download data or bypass licenses. Local JSON/JSONL/CSV exports
must carry an explicit revision, expected row count, source split and scorer.
Revision strings are provenance assertions; validate them against the local
source snapshot. Raw benchmark rows and model outputs are not committed by this
change.

## 5. Training changes: CRAFT-Q and strong content controls

### Phase A — answer compliance and content learning

Before the next large RL run, sweep answer/readout caps on **development data**
and diagnose truncation, parser failures and incorrect content separately. Do not
silently count malformed outputs as correct. Include the original, untouched
scorer in legacy historical comparisons; use the common versioned task-appropriate
scorer in new matched comparisons.

An optional **answer-only SFT warmup** uses only official training targets, no
teacher-generated thought traces. It tests whether final-answer compliance is
limiting reward. It is not SuperCorrect, and its training loss is not evidence of
reasoning quality. SFT reports supervised tokens, not fabricated sampled tokens
or tautological “100% accuracy.” Evaluate a base + SFT-only control separately.

Use RLOO, GRPO-reference and Dr.GRPO-reference with identical LoRA/data/prompt
settings and a quality-only objective first. GRPO-reference includes group
standardization and response-length normalization; Dr.GRPO-reference uses group
centering and a fixed normalizer. These single-update fresh-policy references
have no PPO multi-epoch replay or KL term and are labeled accordingly.

### Phase B — quality-first counterfactual training

`comparison_training.py --algorithm` is selected through its JSON config. The
`craft_quality` arm reuses existing CRAFT sampling and exact branch credit, but
changes the training pool and cost objective. There is a quality-only warmup,
then a predeclared linear cost ramp. For binary correctness r and capped cost c:

```
U = (1 + lambda) r - beta [ r min(c / C, 1) + (1-r) ],  0 <= beta < 1.
```

A wrong answer has constant utility `-beta` regardless of its length. Shortening
an incorrect response alone cannot improve reward. Every individual correct
outcome is preferable to every incorrect outcome at the allowed beta. **This
outcome ordering is not a guarantee of monotonic expected accuracy under SGD.**
The current CRAFT-Q control disables the old task dual multipliers to isolate the
curriculum; original CRAFT's dual-constrained arm remains available unchanged.

The same-prefix estimator remains valid for this transformed outcome reward.
Training logs count all branches and prefills, including branches not used at
deployment. Equal optimizer steps are not compute matching: compare actual
sampled tokens, forward/backward work, GPU time and wall time. The new control
trainer writes fresh adapter checkpoints and refuses output overwrite; it does
**not** support optimizer resume. Warm-starting `--initial-adapter` is an explicit
new run. Original CRAFT atomic resume remains unchanged.

### Phase C — acceptance gates

A promising result must first beat the untrained and SFT-only **always-continue
content controls** on a larger declared development set. Then freeze choices and
measure the accuracy–cost frontier against Thinkless and other strong baselines.
Use at least three separately trained seeds (planned: 7, 17, 27), report each,
and avoid checkpoint selection on test data. The original pilot is descriptive
historical evidence, not a fresh holdout for this design.

Targets in [`study.json`](../experiments/comparison/study.json)—for example +5
accuracy points or 2x lower latency at comparable quality—are goals, not forecasts.
Predeclare a non-inferiority margin before claiming preservation of quality. A
lower confidence bound above the chosen margin is required; a nonsignificant
superiority test does not establish non-inferiority.

## 6. Statistical and cost reporting

The evaluator requires identical case digests, scorer versions and inference
seeds for paired comparisons. Imported author runs are re-scored rather than
trusting a supplied `correct=true`. Every inference error remains in the cohort.
A complete summary is bound to the protocol and prediction-file SHA-256; partial
runs cannot silently enter the comparison table.

Report micro accuracy, task-macro accuracy, per-task effects, exact McNemar tests,
Holm corrections across candidate comparisons, paired group-cluster bootstrap
intervals and a task-cluster sensitivity interval. The current implementation
provides these **within-run** comparisons. It does not pretend that these
intervals measure variation across training seeds; publish all seed results and
perform an explicitly designed seed-level/hierarchical analysis separately.

Latency excludes grading time. Total completion budgets count all candidate,
critique and selection calls. Unknown API token usage stays unknown; reserved
budgets do not turn it into measured zero. Include prefill, hidden-thinking,
failed-request, verifier, CPU tool, embedding, offline data-construction and
training costs. The bridge reports gate prefill even though the gate is not a
sampled generation call. Match GPU type, precision, runtime and load profile.
Metadata agreement is a declaration of matched conditions, not proof of an idle
GPU. Shared-server pilot latency remains observational.

No comparison command emits an automatic SOTA claim. Different base hashes are
marked descriptive. Published-paper accuracy copied into a spreadsheet is not a
reproduction. Report which official recipe, source commit, prompts, sample count,
scorer, model revision and data revision produced every new point.

## 7. Environment compatibility

The successful pilot used PyTorch **2.9.1+cu128**, Transformers **5.5.4**, PEFT
**0.20.0**, and RTX 6000 Ada GPUs. Preserve that working environment. The new local
baseline loader retains the current `Mapping`/`BatchEncoding` fix and handles
multi-token model termination sets and explicit channel parsing. Causal-only
models have a separate local loader, not guessed Gemma internals.

[Environment files](../envs/) separate the existing Gemma stack from mathematical
grading and legacy upstream environments. Math-Verify's inspected pyproject
specified **0.9.0**, `latex2sympy2_extended==1.11.0`, with an explicit ANTLR runtime;
these are source-observed pins, not a claim that this dependency combination was
installed on the user's GPU server. Math scoring fails early if unavailable.

Thinkless's inspected recipe uses Python 3.10/torch 2.4.0 with its own verl fork;
Dr.GRPO's uses vLLM 0.8.4 and oat-llm 0.1.3.post1. Keep these in separate environments
or containers and import outputs. Do not overlay their dependency trees onto the
Gemma service. Never copy upstream instructions that delete a shared `/tmp/ray`;
use an owned runtime directory. No running service or remote machine was modified
by this change. General latest library/model support could not be verified by web
search; use the recorded local environment and startup checks.

## 8. Deliverables and limits

The branch includes a common evaluator, local dataset adapters, strict result
imports, CRAFT comparison bridge, real LoRA reference trainers, CRAFT-Q curriculum,
mathematical loss controls, unit/integration tests, configurations, source registry,
and environment recipes. See [the command guide](COMPARISON_SUITE.md).

Local CPU checks establish implementation properties, not LLM scores. The current
GPU pilot is the only new-model evidence carried forward, and it remains a null
accuracy result. Full upstream baseline executions, new Gemma/Qwen training,
large held-out evaluations and actual multi-seed frontier measurements are still
necessary before saying the project beats these papers.
