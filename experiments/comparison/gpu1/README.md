# Frozen GPU 1 campaign

`campaign-v2.json` contains the actual immutable 61-job plan and input/source
hashes. `data-audit.json` records the official-source split counts and exclusions;
`context-audit.json` records full-cohort prompt length bounds. `preflight.json`
contains engineering validation, not trained-model benchmark results.

The matching source implementation is commit `aa9936749` on
`research/sota-comparison-suite-20260905`. Remote execution is under
`/data/benwulab/gemma4-rl/sota-20260905/campaign-v2/` on `benwulab-remote`.

Inspect `status.json`, `logs/`, and `train/*/metrics.jsonl` for current progress.
The worker writes `REPORT.md` and `aggregate.json` after completed jobs and runs
paired comparisons after full candidate cohorts finish. These files are generated
on the host; this checked-in plan does not imply completion.

See [the campaign guide](../../../docs/GPU1_COMPARISON_CAMPAIGN.md).
