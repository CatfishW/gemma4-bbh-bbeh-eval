# VOLT methodology figure audit

Audit date: 2026-09-02

The figure is based on the `origin/rl-volt` implementation in the remote clone
(`e0dab8a08`) and the run artifacts under `/data/benwulab/gemma4-rl`.

## Code/log checks

- Dataset totals and split: BBH 6,511, BBEH 4,520, USR 1,509; 12,540 records and 60 tasks; calibration indices 0–24, validation 25–49, and test index >=50.
- RL protocol: every fourth task per benchmark is held out (16 tasks); 1,040 train prompts remain after 60 prompts over the 3,072-token cap are dropped; the frozen test has 9,550 examples.
- VOLT loop: 48 iterations, 448 rollouts per iteration, discounted hierarchical Beta posterior (discount 0.92, prior strength 4.0), uncertainty score `sqrt(E[p(1-p)])`, 15% least-recently-sampled exploration floor, and a 1–8 rollout cap.
- Scoring and update: one user message with no system message, binary unchanged exact-match reward, `A = r - b`, token-level REINFORCE with length normalizer 384, AdamW, and LoRA-only updates (rank 32, alpha 64; q/k/v/o plus gate/up/down projections). Length shaping is disabled in the reported run.
- Baseline inset: GRPO uses 56 prompts × 8 fixed rollouts; all-correct/all-wrong groups have zero advantage; 5,432 of 21,504 rollouts have nonzero advantage.
- Selection/evaluation: validation probes every five iterations select checkpoint 45; test decoding is greedy with no allocator/reward at inference, concise_cot max 256 and direct_answer max 64.
- Reported metrics: VOLT concise_cot 3,688/9,550 (38.62%, 127.0 mean completion tokens); GRPO 3,589/9,550 (37.58%, 162.1 tokens); prompt-only CBRR 3,382/9,550 (35.41%).

## Visual QA

The first render repeated the VOLT denominator once. A targeted edit removed the duplicate, and the final 1536×1024 render was inspected at original resolution. The final layout contains the six numbered VOLT stages, the feedback arrow, the protocol split, the GRPO comparator, checkpoint selection, and the metric cards without an invented training or inference step.

The CBRR card is explicitly labeled prompt-only because it comes from the separate API prompt-routing track and is not an apples-to-apples replacement for the local LoRA evaluation.

## Generation record

- Mode: built-in ImageGen, scientific-educational / infographic-diagram, landscape 16:9 (1536×1024).
- Initial prompt: “Create a publication-ready scientific methodology infographic” with the exact protocol, six VOLT stages, GRPO inset, checkpoint rule, and result-card labels listed above.
- Targeted edit prompt: “Edit only the right-side VOLT LoRA result card … remove the duplicated second `(3,688/9,550)`.”
