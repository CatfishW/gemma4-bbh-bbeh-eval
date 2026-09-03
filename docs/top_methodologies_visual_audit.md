# Top-methodologies figures: visual and logic audit

Audit date: 2026-09-03

## Assets

- `top_methodologies_atlas.png`: eight-card overview.
- `prompt_only_methodologies.png`: six prompt-only decision flows.
- `rl_lora_comparison.png`: GRPO versus VOLT LoRA training and evaluation.
- `volt_methodology_figure.png`: detailed VOLT loop from the earlier figure.

All assets were generated with the built-in ImageGen tool in the
`scientific-educational` / `infographic-diagram` mode, saved at 1536×1024, and
inspected at original resolution with the image viewer.

## Logic checks against the repository

- The atlas labels the selected N=8 set: `direct_answer`, `canonical_short`,
  `option_elimination`, `private_verify`, `concise_cot_self_rank_k3`, CBRR,
  GRPO LoRA, and VOLT LoRA.
- Prompt-only diagrams show one user message, frozen weights, answer
  extraction, and the common exact-match scorer. Self-ranking is three samples
  plus a selection pass; CBRR is 25-row calibration followed by one frozen
  policy per task.
- The RL diagram shows 48 iterations, 448 rollouts per iteration, GRPO's
  56×8 fixed groups, VOLT's posterior baseline and uncertainty allocator, the
  15% exploration floor, the 1–8 rollout cap, binary reward, and the correct
  advantage/update rules.
- LoRA details match the run: rank 32, alpha 64, q/k/v/o plus gate/up/down
  projections, and 48.3M trainable parameters (0.94%).
- Evaluation cards match the frozen summaries: Base 19.09%, GRPO 37.58%, VOLT
  38.62%; checkpoint 45; greedy temperature 0; no allocator or reward at
  inference.
- The CBRR comparator is explicitly marked prompt-only and is not presented as
  an apples-to-apples replacement for the local LoRA evaluations.

## Visual corrections

- In the prior VOLT figure, a duplicated `(3,688/9,550)` was removed.
- In the prompt-only figure, a missing space in “reasoning and formatting” was
  corrected.
- In the RL figure, illustrative allocation numbers were removed and replaced
  with the exact range `1–8 rollouts/prompt` so no unlogged example looks like
  a measured result.

No remaining text, metric, arrow, or training/inference boundary was found to
  contradict the code or run logs.
