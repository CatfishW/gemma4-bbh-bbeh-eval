# Data, Model, and Evaluation Statement

## Model

The confirmatory model is `google/gemma-4-E2B-it` at immutable Hub revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`. The downloaded
`model.safetensors` SHA-256 is
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
The Hub cache verifier checked all nine repository files against that revision.

The [official model card](https://huggingface.co/google/gemma-4-E2B-it) describes
E2B as a 35-layer dense model with 2.3B effective parameters (5.1B including
embeddings), a 128K context window, and an Apache 2.0 license. The deployment
uses BF16 weights without quantization, CPU offload, fine-tuning, adapters, or
changes to the downloaded chat template.

The matched exploratory model is `google/gemma-4-E4B-it` at Hub revision
`fee6332c1abaafb77f6f9624236c63aa2f1d0187`; its weight SHA-256 is
`cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`.
The same Hub checksum verification and unmodified BF16 serving conditions apply.

Gemma 4 supports system-role messages, but the registered prompt study and
greedy RL evaluation do not send one. Every request in those protocols has
exactly one `user` message. The public router forwards the original request
bytes and chooses a backend from the existing `model` field; it does not add
messages or prompt text.

A separate post-hoc BBEH reproduction enables Gemma's native thinking through
the pinned tokenizer's `enable_thinking=True` option. The template—not the
caller—then inserts one leading system `<|think|>` turn. That extension uses a
separate output tree and claim; it does not alter the registered results. Its
full protocol and limitations are in
[OFFICIAL_THINKING_EVALUATION.md](OFFICIAL_THINKING_EVALUATION.md).

## Datasets

| Dataset | Revision | Included tasks | Included examples | License |
|---|---|---:|---:|---|
| `suzgunmirac/BIG-Bench-Hard` | `9ee07bd481feebf959a6b59d61ea57bdcf30964d` | 27 | 6,511 | MIT |
| `google-deepmind/bbeh` | `80d12ca916b7158f22293fcf3144f4d3d854d4be` | 23 | 4,520 | Apache 2.0 software; CC BY 4.0 materials |
| `google-deepmind/unpuzzles_and_simple_reasoning` | `39bc520a2f4c243eb04ce1cc27f28c7c61d12e42` | 10 | 1,509 | Apache 2.0 |

USR includes all auto-scorable `simple_reasoning` rows and every non-empty
`original`, `unpuzzle`, or `shifted` target from the unpuzzles files. Empty
targets are excluded rather than imputed. Loader behavior is version controlled
in `eval_benchmarks.py`.

The 12,540 included examples are deterministic source records, not a random
sample. Within each task, source indices 0-24 form calibration, 25-49 form
validation, and indices 50 onward form test. This yields 1,500 calibration,
1,490 validation, and 9,550 test examples. `usr/simple_reasoning/char_ct` has
only 40 rows, which accounts for the ten-row validation difference and gives
that task no test rows.

## Outcome and scoring

The primary outcome is pooled micro exact-match accuracy over the 9,550 E2B
test rows. The scorer extracts recognized JSON/XML answer fields and common
final-answer delimiters, lowercases text, removes limited punctuation/formatting,
normalizes option labels such as `A` and `(A)`, and accepts numerically equal
representations. Both raw and normalized predictions are retained.

Exact match is objective and reproducible but can reject a semantically correct
free-form paraphrase. Benchmark, task, macro-task, latency, token, and error
metrics are therefore reported alongside micro accuracy. No model-based judge
is used for the primary outcome.

## Statistical unit and inference

The example is the paired unit for arm-versus-direct comparisons. The primary
test is exact two-sided McNemar. Secondary finalist comparisons use Holm
family-wise correction. Effect intervals use a paired bootstrap stratified by
benchmark task, preserving each task's sample size. This guards against an arm
appearing strong only because it changed the task mixture.

CBRR observes calibration rewards and known task IDs. Its test claim is
generalization to unseen examples from known tasks. It does not establish
generalization to unseen tasks, domains, or model families.

## Contamination and independence

BBH and related public benchmarks predate the model's January 2025 training-data
cutoff reported by the model card, so training contamination cannot be excluded.
USR supplies a structurally different replication but does not prove data
independence. Prompt selection never reads E2B test labels, yet the base model
may have encountered benchmark material during pretraining.

E4B informed prompt development and is explicitly exploratory. E2B was deployed
only after its arm manifest, split, selection rule, hypotheses, and statistical
analysis were committed. The two models share the Gemma 4 family, so E2B is a
model-scale confirmation rather than a cross-family replication.

## Artifacts and auditability

Each run retains the command, configuration, dataset revisions, prompt template
or policy, per-generation seed, raw response, normalized response, exact-match
decision, token usage, elapsed time, and error text. Selection inputs and
`selection.json` receive SHA-256 digests before test launch. Remote artifacts are
hashed again before transfer and verified before GitHub publication.
