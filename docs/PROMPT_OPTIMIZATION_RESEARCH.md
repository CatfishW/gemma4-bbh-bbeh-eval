# Prompt Optimization Research

## Outcome

The first full challenger established a prompt-only improvement without changing the deployed checkpoint or sending a system message:

| Strategy | BBH | BBEH | USR | Combined |
|---|---:|---:|---:|---:|
| `direct_answer` | 3,198/6,511 | 647/4,520 | 254/1,509 | 4,099/12,540 (32.69%) |
| `private_verify` | 3,516/6,511 | 621/4,520 | 214/1,509 | 4,351/12,540 (34.70%) |

`private_verify` adds 252 correct answers on the full suite. Its gains are concentrated in BBH date understanding, web of lies, colored objects, tables, and object tracking; it regresses on BBEH and several USR negation tasks. That task dependence motivated a calibrated prompt policy instead of a longer universal prompt.

The reward-routed policy uses the first 25 examples of each of 60 tasks as calibration (1,500 examples) and excludes them from its primary score. On the remaining 11,040 examples, replaying the fixed per-task policy over archived model predictions gives:

| Method | Correct | Accuracy | Gain vs direct |
|---|---:|---:|---:|
| `direct_answer` | 3,632/11,040 | 32.90% | - |
| `reward_routed_v1` | 4,738/11,040 | 42.92% | +1,106 / +10.02 points |

The policy keeps `direct_answer` for 30 tasks and assigns one fixed specialized prompt to each of the other 30 tasks. It never selects per example. `scripts/run_reward_routed_policy.sh` repeats the policy online against the deployed API so replay and live inference are recorded separately.

## Research Basis

- [Small Language Models Need Strong Verifiers to Self-Correct Reasoning](https://arxiv.org/abs/2404.17140) finds that small models often need a stronger verifier for reliable correction. This motivated `selective_verify`, which changes an initial answer only after finding a concrete contradiction.
- [Sample, Scrutinize and Scale](https://arxiv.org/abs/2502.01839) reports that comparing candidate responses gives a useful verification signal and that reasoning and verification benefit from different output styles. This motivated `compare_then_commit` and answer-only final output.
- [When To Solve, When To Verify](https://arxiv.org/abs/2504.01005) finds that solution sampling is usually more compute-efficient than heavy generative verification at practical budgets. This motivated the one-pass `fast_slow_gate` rather than a deep verifier loop.
- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) demonstrates reward-driven instruction search, while [Revisiting OPRO](https://arxiv.org/abs/2405.10276) finds that small models are weak self-optimizers and recommends direct, explicit instructions. The new candidates therefore remain short and are selected externally from exact-match rewards.
- [RLPrompt](https://arxiv.org/abs/2205.12548) formulates discrete prompt selection as reinforcement learning and emphasizes reward stabilization. `reward_routed_v1` is a lightweight contextual-bandit analogue: task ID is the context, prompt strategy is the arm, exact-match correctness is the binary reward, and a Beta(1,1) posterior stabilizes arm estimates.
- [Rethinking Prompt Optimization](https://arxiv.org/abs/2507.09839) argues for preserving successful prompt components as positive reinforcement. The policy preserves direct answer on ties and requires a challenger to earn at least one additional calibration reward.

## New Strategies

| Strategy | Mechanism |
|---|---|
| `selective_verify` | Check a bounded list of likely errors and revise only on a specific contradiction. |
| `compare_then_commit` | Contrast the two strongest candidates against the exact constraints. |
| `fast_slow_gate` | Keep a clear direct answer; spend one verification pass only when needed. |
| `constraint_guard` | Test the candidate against each decisive constraint once. |
| `negation_label_guard` | Protect negations, quantifiers, and content-to-option-label mapping. |
| `draft_verify` | Combine a terse private draft with one targeted final check. |

All strategies are appended to the benchmark item inside the sole `user` message. The harness records `system_messages_sent: 0`, and the Nginx proxy only rewrites the URL path.

## RL Decision

Weight-level RL was considered but is not mixed into this comparison. A GRPO/RLPrompt-style adapter would change the model under test, require a separately versioned checkpoint, and need a new train/validation/test split to avoid optimizing directly on benchmark answers. It is justified only after the prompt-only policy is confirmed online and after a disjoint training corpus is chosen. The contextual bandit captures the useful reward-allocation idea now while leaving `/data/models/gemma-4-E4B-it` and the deployed `SubTokenLLM` weights unchanged.
