# CRAFT experimental configurations

Read [the method, derivation, commands and limitations](../../docs/CRAFT_RL.md).

The [two-GPU implementation pilot](PILOT_20260905.md) provides a reproducible
short-context study runner, declared task selection, six validation controls,
and strict paired result aggregation.
See the [completed GPU results](../../docs/CRAFT_GPU_RESULTS.md): equal sampled-gate
accuracy to the untrained model on 68 validation rows, with fewer generated tokens;
no demonstrated accuracy improvement.

These configurations define the full method and four ablations. They do not
contain results, target labels, learned weights, or fitted accuracy floors.
`calibrate-targets` constructs model/data-bound task floors from calibration only.
Training must run in the existing, working `rl-volt` Gemma/PEFT environment.

```bash
python -m unittest discover -s tests -p 'test_craft_*.py' -v
python -m rl_craft smoke --output /tmp/craft-smoke --iterations 12
```

The smoke backend is a tiny CPU LoRA model, not Gemma. Equal iteration counts in
these configs are NOT compute matching; use actual sampled tokens, prefills,
GPU time, wall time and accuracy for comparisons.
