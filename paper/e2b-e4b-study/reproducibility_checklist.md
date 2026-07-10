# Reproducibility Checklist

- [x] Exact model repository, revision, and weight SHA-256 recorded.
- [x] Exact dataset revisions and deterministic source-index splits recorded.
- [x] Protocol committed before E2B benchmark inference.
- [x] All prompt templates and inference configurations version controlled.
- [x] Per-example seeds, raw predictions, normalized predictions, usage, latency, and errors retained.
- [x] Finalists selected automatically without E2B test-label access.
- [x] Paired exact tests, family-wise correction, and stratified bootstrap intervals reported.
- [x] Direct answer and primary policy repeated at two additional seeds.
- [x] Matched E4B screening labeled exploratory rather than confirmatory.
- [x] Question, ground truth, direct answer, routed answer, wins, and losses included.
- [x] No system-role messages or gateway system-prompt injection.
- [x] Negative results and validity limitations retained.
