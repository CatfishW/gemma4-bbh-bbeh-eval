# Reward-Routed v2 Examples

All examples are from the held-out suffix (index >= 25). Prompt arms were fixed from the excluded 25-example calibration prefix.

| Selected arm | Held-out correct | Direct correct | Delta |
|---|---:|---:|---:|
| `canonical_short` | 347/826 | 333/826 | +14 |
| `compare_then_commit` | 713/1445 | 499/1445 | +214 |
| `constraint_guard` | 169/320 | 157/320 | +12 |
| `direct_answer` | 2429/7353 | 2426/7353 | +3 |
| `draft_verify` | 137/296 | 99/296 | +38 |
| `private_verify` | 216/625 | 115/625 | +101 |
| `selective_verify` | 9/175 | 3/175 | +6 |

## New-arm win: bbh/date_understanding #149

- Selected strategy: `compare_then_commit`
- Ground truth: `(D)`
- Task held-out delta vs direct: `+108`
- Direct correct: `false`
- Policy correct: `true`

Question:

```text
This is the last day of 1899. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/11/1899
(B) 12/21/1887
(C) 12/24/1899
(D) 12/21/1899
(E) 03/22/1900
```

Direct answer:

```text
12/21/1899
```

Policy answer:

```text
(D)
```

## New-arm win: bbh/date_understanding #97

- Selected strategy: `compare_then_commit`
- Ground truth: `(E)`
- Task held-out delta vs direct: `+108`
- Direct correct: `false`
- Policy correct: `true`

Question:

```text
Tomorrow is 11/12/2019. What is the date today in MM/DD/YYYY?
Options:
(A) 11/12/2019
(B) 11/11/2042
(C) 11/11/2020
(D) 11/01/2019
(E) 11/11/2019
(F) 11/25/2019
```

Direct answer:

```text
11/11/2019
```

Policy answer:

```text
(E)
```

## New-arm win: bbh/penguins_in_a_table #112

- Selected strategy: `draft_verify`
- Ground truth: `(A)`
- Task held-out delta vs direct: `+39`
- Direct correct: `false`
- Policy correct: `true`

Question:

```text
Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  What is the shortest height?
Options:
(A) 50
(B) 80
(C) 60
(D) 70
(E) 90
```

Direct answer:

```text
50
```

Policy answer:

```text
(A)
```

## New-arm win: bbh/penguins_in_a_table #118

- Selected strategy: `draft_verify`
- Ground truth: `(B)`
- Task held-out delta vs direct: `+39`
- Direct correct: `false`
- Policy correct: `true`

Question:

```text
Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  How many penguins are less than 8 years old?
Options:
(A) 1
(B) 2
(C) 3
(D) 4
(E) 5
```

Direct answer:

```text
2
```

Policy answer:

```text
(B)
```

## New-arm win: bbh/ruin_names #248

- Selected strategy: `constraint_guard`
- Ground truth: `(D)`
- Task held-out delta vs direct: `+11`
- Direct correct: `false`
- Policy correct: `true`

Question:

```text
Which of the following is a humorous edit of this artist or movie name: 'toto'?
Options:
(A) thto
(B) tioto
(C) totco
(D) tote
```

Direct answer:

```text
(B)
```

Policy answer:

```text
(D)
```

## New-arm win: bbh/ruin_names #55

- Selected strategy: `constraint_guard`
- Ground truth: `(A)`
- Task held-out delta vs direct: `+11`
- Direct correct: `false`
- Policy correct: `true`

Question:

```text
Which of the following is a humorous edit of this artist or movie name: 'rocky'?
Options:
(A) ricky
(B) rotky
(C) tocky
(D) crocky
```

Direct answer:

```text
(B)
```

Policy answer:

```text
(A)
```

## Policy loss: bbeh/bbeh_disambiguation_qa #43

- Selected strategy: `compare_then_commit`
- Ground truth: `(E)`
- Task held-out delta vs direct: `-3`
- Direct correct: `true`
- Policy correct: `false`

Question:

```text
Here is a sentence with pronoun(s) whose antecedent(s) can either be derived from the context or is ambiguous.
Sentence: The principal, the teacher and the student’s parents met to discuss his declining grades. She suggested a tutor and the parents scheduled weekly classes for the following weeks hoping it would improve his performance. They were very satisfied with the final result.
Which of the following options correctly explains the antecedent(s) of the pronoun(s)?
(A) The principal suggested a tutor. The parents scheduled a meeting.
(B) The teacher suggested a tutor. The parents scheduled a meeting.
(C) The parents scheduled a meeting. The parents were satisfied with the result.
(D) The parents scheduled a meeting. The tutor and the teacher were satisfied with the results.
(E) It is ambiguous who suggested a tutor. It is ambiguous who was satisfied with the result.
(F) None of the above.
```

Direct answer:

```text
(E)
```

Policy answer:

```text
(C)
```

## Policy loss: bbeh/bbeh_disambiguation_qa #52

- Selected strategy: `compare_then_commit`
- Ground truth: `(F)`
- Task held-out delta vs direct: `-3`
- Direct correct: `true`
- Policy correct: `false`

Question:

```text
Here is a sentence with pronoun(s) whose antecedent(s) can either be derived from the context or is ambiguous.
Sentence: An employee asked the manager if it is possible to share his document.
Which of the following options correctly explains the antecedent(s) of the pronoun(s)?
(A) The employee is a man. The manager is a man. The employee is seeking access to the manager’s document.
(B) The employee is a man. The manager is a woman. The employee seeks advice about sharing his own document.
(C) The employee is a woman. The manager is a man. The employee is seeking access to the manager’s document.
(D) The employee is a woman. The manager is a man. The employee is seeking access to the manager’s document.
(E) The employee’s gender is ambiguous. The manager’s gender is ambiguous. The document belongs to the employee.
(F) The employee’s gender is ambiguous. The manager’s gender is ambiguous. The owner of the document is ambiguous.
```

Direct answer:

```text
(F)
```

Policy answer:

```text
(D)
```

## Policy loss: bbeh/bbeh_disambiguation_qa #53

- Selected strategy: `compare_then_commit`
- Ground truth: `(E)`
- Task held-out delta vs direct: `-3`
- Direct correct: `true`
- Policy correct: `false`

Question:

```text
Here is a sentence with pronoun(s) whose antecedent(s) can either be derived from the context or is ambiguous.
Sentence: The patient, who is a teacher, was referred to the specialist because they are an expert on rare skin conditions. She struggled to get her degree.
Which of the following options correctly explains the antecedent(s) of the pronoun(s)?
(A) The specialist is a new teacher and the patient had a referral.
(B) The patient is a teacher and the specialist has a rare skin condition.
(C) The specialist struggled to get her degree and the patient has a skin condition.
(D) The patient struggled to get her degree and struggled to get an appointment.
(E) None of the above.
```

Direct answer:

```text
(E)
```

Policy answer:

```text
(C)
```

## Policy loss: bbeh/bbeh_disambiguation_qa #72

- Selected strategy: `compare_then_commit`
- Ground truth: `(E)`
- Task held-out delta vs direct: `-3`
- Direct correct: `true`
- Policy correct: `false`

Question:

```text
Here is a sentence with pronoun(s) whose antecedent(s) can either be derived from the context or is ambiguous.
Sentence: The teacher met the engineer at the library where she was researching renewable energy and he was reading about marine animals. Afterwards, they walked to a nearby park where she explained that they were excited about the potential of wind power. He had some reservations.
Which of the following options correctly explains the antecedent(s) of the pronoun(s)?
(A) The teacher was researching renewable energy and the engineer was reading about marine animals.
(B) The engineer was researching renewable energy and the teacher was reading about marine animals.
(C) The teacher was excited about wind power and the engineer had some reservations.
(D) The engineer was excited about wind power and the teacher had some reservations.
(E) Ambiguous.
```

Direct answer:

```text
(E)
```

Policy answer:

```text
(A)
```
