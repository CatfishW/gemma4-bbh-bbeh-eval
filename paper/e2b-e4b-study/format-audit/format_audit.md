# Strict JSON Format Audit

Raw validity requires exactly one JSON object with one string `answer` field and no Markdown fence. Recoverable validity permits one surrounding code fence.

| Run | Answer accuracy | Raw schema | Recoverable schema | Fenced | Mean tokens | Errors |
|---|---:|---:|---:|---:|---:|---:|
| `E2B` | 8.19% | 4.72% | 39.36% | 1098/2990 | 64.50 | 0 |
| `E4B` | 12.24% | 9.87% | 26.79% | 570/2990 | 72.81 | 0 |

## Invalid examples

### E2B: bbh/boolean_expressions #5

- Ground truth: `True`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```

### E2B: bbh/boolean_expressions #1

- Ground truth: `True`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```

### E2B: bbh/boolean_expressions #2

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E2B: bbh/boolean_expressions #0

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E2B: bbh/boolean_expressions #4

- Ground truth: `True`
- Answer correct: `False`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E2B: bbh/boolean_expressions #3

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E2B: bbh/boolean_expressions #6

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E2B: bbh/boolean_expressions #7

- Ground truth: `True`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```

### E4B: bbh/boolean_expressions #7

- Ground truth: `True`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```

### E4B: bbh/boolean_expressions #3

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E4B: bbh/boolean_expressions #0

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E4B: bbh/boolean_expressions #4

- Ground truth: `True`
- Answer correct: `False`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E4B: bbh/boolean_expressions #2

- Ground truth: `False`
- Answer correct: `False`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```

### E4B: bbh/boolean_expressions #6

- Ground truth: `False`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "False"
}
` ` `
```

### E4B: bbh/boolean_expressions #9

- Ground truth: `True`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```

### E4B: bbh/boolean_expressions #12

- Ground truth: `True`
- Answer correct: `True`
- Raw response:

```text
` ` `json
{
  "answer": "True"
}
` ` `
```
