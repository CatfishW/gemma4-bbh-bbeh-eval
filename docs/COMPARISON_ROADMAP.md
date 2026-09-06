# Paper-grounded comparison roadmap

Status: research planning update, 2026-09-05. This branch starts from the completed CRAFT GPU pilot at ba2977c79d11229f63c08cbfa1cffcb80d17d2b8. The pilot artifacts remain unchanged.

## Closest comparison and novelty correction

[Thinkless](https://github.com/VainF/Thinkless) already learns short/long reasoning with separate control-token and response losses. CRAFT should not claim adaptive thinking or separated controller/content losses as new. Its narrower hypothesis is same-prefix counterfactual sampling with exact shared-prefix/gate/suffix credit and importance-corrected allocation. Compare both the upstream Thinkless recipe and explicitly labeled matched-base transfers.

## Required baseline groups

- Frozen prompting: CoT, self-consistency, [Chain of Draft](https://github.com/sileix/chain-of-draft), [Self-Refine](https://github.com/madaan/self-refine), planning, and correctly configured native thinking.
- Modules: [PAL](https://github.com/reasoning-machines/pal), [Program of Thoughts](https://github.com/TIGER-AI-Lab/Program-of-Thoughts) (TMLR 2023), [Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm), and [Buffer of Thoughts](https://github.com/YangLing0818/buffer-of-thought-llm) (NeurIPS 2024 Spotlight).
- Training: RLOO, GRPO/Dr.GRPO, [Thinkless](https://github.com/VainF/Thinkless), [ThinkPrune](https://github.com/UCSB-NLP-Chang/ThinkPrune), [s1](https://github.com/simplescaling/s1), [SuperCorrect](https://github.com/YangLing0818/SuperCorrect-llm) (ICLR 2025), and [ReasonFlux-PRM](https://github.com/Gen-Verse/ReasonFlux) (NeurIPS 2025).

Conference labels above are explicit in inspected upstream artifacts where supplied; this is not a verified global September 2026 SOTA ranking. General live web search was unavailable.

## Revised experiment direction

The completed pilot matched base sampled accuracy at 15/68 while saving tokens; it did not demonstrate an accuracy gain. Prioritize answer compliance and content learning on official GSM8K/MATH training sources before optimizing stopping. Use quality-only warmup, followed by bounded success-conditioned cost shaping: failed answers should not become more rewarding merely by being shorter. Keep SFT-only, quality-only RL, always-continue, and sampled-action controls.

Expand evaluation to full BBH/BBEH/USR, GSM8K, MATH-500, arithmetic perturbation suites, MMLU-Pro and GPQA, with AIME as a small-N stress test and Game24 only under its real task verifier. Keep source-puzzle variants together and reject official test-to-train relabeling. Existing exposed test sets are not new confirmatory holdouts.

Use identical case identities, model/runtime/precision, task-appropriate graders, total inference budgets, and multiple training seeds. Count all candidate, critique, verifier, embedding and tool costs. Re-score imported upstream predictions; forbid label-triggered retries. Report paired group/task uncertainty and preserve all error cases. Distinguish a transferred prompt from a faithful author-recipe reproduction.

Preserve the working Gemma pilot environment (torch 2.9.1+cu128, transformers 5.5.4, peft 0.20.0). Isolate older Thinkless/verl and Dr.GRPO/oat/vLLM environments rather than overlaying their dependencies onto running services.

## Code publication status

A local comparison implementation, tests, source registry, dataset/config manifests and environment recipes were prepared in this work session. The GitHub connector blocked the code-blob upload twice because it could not determine the safety status. Therefore this commit publishes the roadmap only: it does not claim the new implementation or its CI workflow is present on this branch. The separately delivered patch/ZIP contains the proposed code. No new model-training results or SOTA claim is made.
