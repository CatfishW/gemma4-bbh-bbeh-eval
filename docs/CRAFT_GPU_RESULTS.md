# CRAFT: completed two-GPU pilot (2026-09-05)

CRAFT completed real Gemma 4 E2B LoRA training on both RTX 6000 Ada GPUs. On the declared 68-row validation pilot, its sampled gate matched the untrained model at **15/68 (22.06%)** and used **21.6% fewer generated tokens**. This run does **not** establish an accuracy improvement or non-inferiority. Forced continuation with the untrained model had the highest observed score, 17/68.

## Validation results

Every condition completed exactly the same 68 examples. All had zero response-parser failures. Latencies are observations from a shared server, not isolated performance measurements.

| Model | Gate | Correct / 68 | Accuracy | Mean tokens | Mean seconds | Stop rate | Truncated answers |
|---|---|---:|---:|---:|---:|---:|---:|
| Untrained E2B | Sampled | 15 | 22.06% | 96.0 | 3.643 | 42.6% | 11 |
| Untrained E2B | Always continue | 17 | 25.00% | 135.8 | 5.081 | 0.0% | 11 |
| CRAFT | Sampled | 15 | 22.06% | 75.3 | 2.833 | 63.2% | 9 |
| CRAFT | Always continue | 14 | 20.59% | 134.6 | 5.015 | 0.0% | 9 |
| No crossfit | Sampled | 13 | 19.12% | 75.7 | 2.887 | 63.2% | 9 |
| No crossfit | Always continue | 16 | 23.53% | 135.1 | 5.005 | 0.0% | 9 |

The five task families excluded from training contributed 20 validation rows. **All six conditions scored 3/20** on those rows. The three sampled-gate conditions scored 12/48 (base), 12/48 (CRAFT), and 10/48 (no crossfit) on the training task families’ validation rows.

## Paired comparisons

Positive differences favor CRAFT with its sampled gate. Intervals resample examples within the 17 fixed tasks (10,000 replicates); they do not measure variation across training seeds or new task families. These are exploratory, unadjusted comparisons.

| Comparator | CRAFT-only wins / losses | Accuracy difference | Within-task bootstrap 95% | Exact McNemar p |
|---|---:|---:|---:|---:|
| Untrained sampled gate | 1 / 1 | +0.00 pp | [-2.94, +2.94] pp | 1.000 |
| No-crossfit sampled gate | 2 / 0 | +2.94 pp | [+0.00, +7.35] pp | 0.500 |
| CRAFT always continue | 1 / 0 | +1.47 pp | [+0.00, +4.41] pp | 1.000 |

CRAFT saved 44.1% of generated tokens versus its own forced-continuation control. The one additional correct answer is insufficient evidence that stopping improves accuracy. Its forced-continuation score fell from the untrained model’s 17/68 to 14/68, which also argues against claiming a general reasoning gain.

## Training and implementation checks

Both runs used a 300-row calibration pool (12 task families), sampled 48 root episodes with replacement, and visited 46 distinct questions. Neither used validation labels for training or checkpoint selection. The fixed final checkpoint was step 12. The only algorithm difference between arms was the suffix baseline; exact settings and selection rules are in the [pilot protocol](../experiments/craft/PILOT_20260905.md).

| Arm | Updates | Sampled tokens | Generation prefill tokens | Step wall seconds | Peak PyTorch allocation |
|---|---:|---:|---:|---:|---:|
| CRAFT | 12 | 12,028 | 109,471 | 493.9 | 14.81 GB |
| No crossfit | 12 | 12,031 | 109,442 | 495.1 | 14.81 GB |

Training plus checkpoint writing took 503.3 seconds for CRAFT and 504.3 seconds for no crossfit, excluding model loading and validation. Sampled-token use happened to differ by only three tokens; this is an observed near match, not an enforced total-compute match. Each arm retained all 13 atomic checkpoints, including its initial untrained adapter.

- Fixed a real startup failure: Transformers 5.5.4 returns `BatchEncoding` from `apply_chat_template`; accepting `Mapping` instead of only `dict` lets the backend read token IDs correctly.
- Updated the generation self-check to put its output options in `GenerationConfig`, removing the installed API’s deprecation warning.
- Added per-step GPU memory and evaluation truncation, parser and stop-probability metrics.
- Added the deterministic cohort builder, independent GPU workers, strict complete-result aggregation and regression tests.
- All **109 repository tests passed** on the remote environment. GPU generation/log-probability self-checks, finite-gradient LoRA updates, checkpoint integrity and completed-run reload/resume checks passed for both arms. The initial adapters were byte-identical; all 410 adapter tensors changed during training. Base parameters remained frozen by construction; a full in-memory post-training base-byte comparison was not performed.

Runtime: PyTorch 2.9.1+cu128, Transformers 5.5.4, PEFT 0.20.0, two 48 GB RTX 6000 Ada GPUs, NVIDIA driver 570.207. Existing GPU services continued running. No service was stopped and no sudo action was needed.

## Scope and next experiment

This was a one-seed implementation pilot with 12 updates, a 2,048-token context limit, and short-context task selection. It is not the full BBH/BBEH/USR validation set, a frozen-test result, or a comparison with GRPO/VOLT. The observed token savings do not establish accuracy non-inferiority or generalization to unseen families.

Seven of CRAFT’s nine answer truncations occurred on USR logic tasks. The calibration targets for the selected training logic task are one token, so the cap is not inherently too short for a correct answer. Before scaling training, test final-answer compliance and answer budgets on calibration data, then run more training seeds and a larger declared validation cohort with measured total compute. Keep the 50+ test rows excluded from this development process.

## Artifacts

- [Aggregate results and paired statistics](../results/craft-pilot-20260905/comparison.json)
- [Model/source hashes and hardware provenance](../results/craft-pilot-20260905/provenance.json)
- [Exact CRAFT protocol](../results/craft-pilot-20260905/full-protocol.json) and [ablation protocol](../results/craft-pilot-20260905/no-crossfit-protocol.json)
- [Test output](../results/craft-pilot-20260905/tests.log)
- Implementation commit: `bdb8392e2`, built on `593fc44559dca59cc3d31994588fe2c5557ae103`.

Raw data, traces, predictions, optimizer states and adapters remain on `benwulab-remote` under `/data/benwulab/gemma4-rl/craft-20260905/pilot/`. Final adapters are `full/checkpoint-00012/adapter/` and `no_crossfit/checkpoint-00012/adapter/`. The experiment source is the sibling `repo/` directory; the report's relative checkpoint paths resolve from that directory.
