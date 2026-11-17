# Stage 0: open models on the R set

R: tasks of "reasoning over a card", the card in the prompt, two worked examples, greedy decoding. Commands are checked by running them in a sandbox (Linux) or by normalised match (macOS, Windows, system-state commands). Share of correct answers, percent.

| Model | Params | slot (128) | os (59) | choose (28) | none (16) | react (25) | fix (19) | two (27) | arith (40) | direct (20) | All | 95 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Copy the card's first example (floor) |  | 0 | 10 | 0 | 0 | 20 | 0 | 0 | 0 | 25 | **4.4** (16/362) | 3–7 |
| SmolLM2-135M-Instruct | 135M | 9 | 15 | 0 | 100 | 20 | 0 | 0 | 5 | 25 | **13.5** (49/362) | 10–17 |
| SmolLM2-360M-Instruct | 362M | 45 | 37 | 29 | 38 | 20 | 42 | 15 | 8 | 60 | **34.8** (126/362) | 30–40 |
| Boris-2-Think | 144M | 0 | 8 | 0 | 31 | 28 | 0 | 0 | 2 | 15 | **5.8** (21/362) | 4–9 |
| Reference answers (grader self-test) |  | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | **100.0** (362/362) | 99–100 |

Categories: **slot** — one card: find the example, fill the values; **os** — one card with a section per OS: take the right one; **choose** — three cards: take the right one; **none** — three cards, none fits: say NONE; **react** — rules + the last output: the next action; **fix** — a failed command and its error: the corrected one; **two** — a two-step command from two cards; **arith** — a number read or computed from an output; **direct** — one card, nothing to fill.

## Where the answers go wrong

| Failure | SmolLM2-135M-Instruct | SmolLM2-360M-Instruct | Boris-2-Think |
|---|---|---|---|
| placeholder left unfilled | 103 | 15 | 30 |
| another program | 55 | 52 | 211 |
| right program, wrong arguments | 67 | 95 | 27 |
| said NONE | 28 | 7 | 5 |
| wrong label or number | 58 | 67 | 68 |
| empty answer | 2 | 0 | 0 |
| *wrong in all* | 313 | 236 | 341 |

Boris-2-Think wrote a `<think>` block in 362 of 362 answers and a `Final answer:` line in 285.
