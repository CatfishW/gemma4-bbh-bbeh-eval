# E2B Direct-Fallback Replay Sensitivity

This is a post-hoc sensitivity analysis, not a preregistered primary test. It replaces CBRR rows assigned to `direct_answer` with the corresponding 64-token-cap baseline output while retaining frozen CBRR outputs for specialized prompt assignments.

- Direct baseline: 2520/9550 (26.39%).
- Registered CBRR: 3382/9550 (35.41%).
- Fallback replay: 3381/9550 (35.40%), +9.02 points versus direct.
- Correct-answer change from registered CBRR: -1.
- Mean completion tokens after replay: 49.86.

This isolates the direct-fallback cap only. It does not make every specialized arm token-matched; the registered CBRR token and cap audit remains the governing cost report.
