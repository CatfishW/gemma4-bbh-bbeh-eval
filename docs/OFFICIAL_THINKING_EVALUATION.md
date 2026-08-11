# Gemma 4 native-thinking BBEH evaluation

<p align="center">
  <a href="./OFFICIAL_THINKING_EVALUATION.md"><img src="https://img.shields.io/badge/Language-English-0969DA?style=for-the-badge" alt="English"></a>
  <a href="./OFFICIAL_THINKING_EVALUATION.zh-CN.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-DE2910?style=for-the-badge" alt="简体中文"></a>
</p>

## Why this is a separate evaluation

Gemma 4 Technical Report Table 5 reports a 21.9% BBEH micro average for E2B
and says that all models in the table use thinking unless explicitly stated.
The repository's original frozen RL evaluation deliberately used no system
message, greedy decoding, and 64/256-token limits. Those results remain valid
for that registered deployment protocol, but comparing them directly with
21.9% would confound model quality with prompting and inference mode.

The native-thinking profile is a post-hoc protocol correction, not a rewrite
of the registered study. It writes to a separate output tree and has two aims:

1. run the unmodified base E2B checkpoint on all 4,520 BBEH rows for the only
   direct comparison with the paper;
2. compare Base, GRPO, and VOLT on the 3,370 frozen rows with index 50 or higher
   under the same public native-thinking profile.

## Pinned v3 protocol

| Component | Fixed value |
|---|---|
| Base model | `google/gemma-4-E2B-it` revision `70af34e20bd4b7a91f0de6b22675850c43922a03` |
| Precision | BF16, no quantization |
| BBEH | revision `80d12ca916b7158f22293fcf3144f4d3d854d4be` |
| Prompt | `task.json` input plus the exact BBEH Appendix-C evaluation suffix |
| Messages | one `user` message; no manually supplied system message |
| Thinking | `apply_chat_template(..., enable_thinking=True)` |
| Sampling | one sample, temperature 1.0, top-p 0.95, top-k 64 |
| Output ceiling | 8,192 new tokens |
| Seed | `20260709`, with the effective seed recorded per batch |
| Parsing | pinned tokenizer's `parse_response`; score `content` only |
| Scoring | pinned upstream `bbeh/bbeh/evaluate.py` |
| Failure accounting | overlong prompts abort; parse failures score empty; truncations are counted |

The 8,192-token ceiling is intentionally prominent. Google does not disclose
the BBEH ceiling, seed, sample count, or complete internal harness used for
Table 5. Therefore even a matching score would be a best-public reproduction,
not proof of implementation identity.

## Exact prompt and output contract

The evaluator sends the BBEH input followed by the suffix published by the
benchmark authors:

```text
Think step by step, and when you provide the final answer, please use the prefix
"The answer is:" without any modification, and provide the answer directly, with
no formatting, no bolding, and no markup. ...
```

It passes that text as one user message with `enable_thinking=True`. The pinned
Gemma tokenizer renders one system thinking-control turn automatically:

```text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
... BBEH prompt and suffix ...<turn|>
<|turn>model
```

A conforming generated response looks like:

```text
<|channel>thought
... private reasoning ...<channel|>The answer is: disproved<turn|>
```

`decode(..., skip_special_tokens=True)` must not be used for scoring because it
concatenates the reasoning and final answer. Instead, `parse_response` yields:

```json
{
  "thinking": "... private reasoning ...",
  "content": "The answer is: disproved"
}
```

Only `content` is passed to the official BBEH scorer. The reasoning, final
content, raw control-token response, token counts, stop token, truncation flag,
parser error, and exact-match decision are retained in `predictions.jsonl`.

An initial calibration-only smoke test without the published BBEH suffix was
intentionally invalidated: Gemma put a second long explanation in the final
channel, and the official scorer could not reliably extract the answer. No
full or frozen result was observed under that draft. A later 40-row,
length-sorted engineering run with the correct suffix was also halted without
a summary so v3 could add weight/config/code hashes and exclusive output
locking. Neither partial run is reportable. The v3 scientific protocol was
fixed before its clean full run began.

## Cells and statistical comparison

| Cell | Rows | Permitted claim |
|---|---:|---|
| Base, all BBEH | 4,520 | descriptive comparison with the paper's 21.9% |
| Base, filtered from all-row predictions | 3,370 | frozen baseline |
| GRPO checkpoint 45 | 3,370 | frozen compute-matched RL baseline |
| VOLT checkpoint 45 | 3,370 | frozen new-method result |

After all cells finish, the comparison script checks exact key equality and
denominators, then reports micro accuracy, mean completion/thinking tokens,
truncation and parser failures, paired wins/losses, exact two-sided McNemar
tests, task-stratified bootstrap intervals, and completion-token changes.
Provisional length-sorted batches are not interpreted as interim accuracy.

## Running and resuming

```bash
# Runs Base/all, GRPO/frozen, VOLT/frozen, then paired analysis on logical GPU 0.
CUDA_VISIBLE_DEVICES=0 ./scripts/run_official_thinking_evals.sh
```

Each cell appends and fsyncs complete batches. If interrupted, rerunning the
shell script adds `--resume`, verifies the full immutable configuration, and
generates only missing prompt IDs. It refuses mixed revisions, duplicate IDs,
changed protocols, silent prompt skipping, and output directories that were
not explicitly resumed.

The remote run writes beneath:

```text
/data/benwulab/gemma4-rl/runs/evals-official-thinking/v3/
  base-all/
  grpo-frozen-test/
  volt-frozen-test/
  comparison.json
  comparison.md
```

The version-controlled specification is
[`experiments/rl/official_thinking_e2b_bbeh.json`](../experiments/rl/official_thinking_e2b_bbeh.json),
and the implementation is
[`rl/eval_official_thinking.py`](../rl/eval_official_thinking.py).
