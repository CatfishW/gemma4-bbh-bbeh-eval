# Frozen inference development fixtures

See [the implementation and evaluation guide](../../docs/ANSWER_STABLE_INFERENCE.md).

- `skills.json` is a manually authored, development-tested arithmetic microprogram
  example. It is not a benchmark-derived answer cache or a pretrained skill bank.
- `smoke.jsonl` is unscored synthetic input. Three inputs have exact/template fast
  paths; one deliberately requires a model and remains unanswered in offline mode.
- Unit/property tests and neural-utility tests live in `tests/test_frozen_*.py`.

Use `python -m frozen_inference --help`. No published result files or original
protocols are modified. No real-model accuracy or speedup is claimed.
