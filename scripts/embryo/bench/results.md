# Embryo-56M on the KLM-1.6 benchmarks

Zero-shot, length-normalized accuracy (acc_norm), percent. ARC / HellaSwag / PIQA: lm-eval 0.4.12; ArithMark-3: the official script's data and scoring rule. Embryo-56M: backbone of `final5-release/model.cmf` (q8_2f), raw text, through `cortiq loglik`. Other rows: Open SLM leaderboard (AxiomicLabs/Open_SLM_Leaderboard, read 2026-11-10), reference excerpt with attribution; Index recomputed from those rows by the Space's stated formula.

| Model | Params | ARC-Easy | ARC-Challenge | HellaSwag | PIQA | ArithMark-3 | Int Index |
|---|---|---|---|---|---|---|---|
| *chance* | | 25.00 | 25.00 | 25.00 | 50.00 | 25.00 | 0.00 |
| **Embryo-56M** | 55.9M | 32.95 | 21.42 | 26.83 | 54.30 | 36.40 | 6.53 |
| KLM-1.6 | 99.0M | 61.07 | 33.53 | 45.28 | 70.18 | 55.90 | 33.95 |
| KLM-1.5 | 99.0M | 59.68 | 32.68 | 43.74 | 68.55 | 48.20 | 30.26 |
| GPT-X3-150M | 149.0M | 59.64 | 31.31 | 46.16 | 71.33 | 50.90 | 33.05 |
| SmolLM2-135M | 135.0M | 58.63 | 29.69 | 43.22 | 68.44 | 39.20 | 27.13 |
| LFM-2.5-230M-Base | 229.7M | 59.72 | 34.98 | 39.29 | 63.38 | 36.60 | 23.47 |
| void.1 | 90.1M | 49.32 | 29.35 | 39.46 | 67.19 | 43.70 | 24.38 |
| *similar size* | | | | | | | |
| Supra-50M-Base | 51.8M | 45.88 | 25.00 | 31.83 | 62.51 | 36.30 | 15.85 |
| Nexus-Erebus-50M | 54.1M | 44.40 | 22.70 | 27.05 | 58.27 | 33.10 | 10.33 |
| Pythia-70M | 70.0M | 31.65 | 23.63 | 27.49 | 53.48 | 27.00 | 4.26 |
| Pythia-31M | 31.0M | 33.88 | 21.67 | 27.14 | 56.26 | 34.90 | 7.58 |
| michel-tiny | 55.7M | 37.37 | 21.76 | 28.15 | 57.34 | 28.70 | 7.72 |

## Rank by Intelligence Index

| Size class | Embryo-56M rank | Models in class | Leader | Leader index | Neighbours (index) |
|---|---|---|---|---|---|
| < 100M | 131 | 198 | KLM-1.6 | 33.95 | tinyLM-8M-exp (6.82), Rambley-V2-Delta-10M (6.70), ZeroS-Micro-v1.0 (6.49), Atom 3.4M (6.37) |
| < 150M | 174 | 245 | KLM-1.6 | 33.95 | tinyLM-8M-exp (6.82), Rambley-V2-Delta-10M (6.70), ZeroS-Micro-v1.0 (6.49), Atom 3.4M (6.37) |
| < 250M | 184 | 255 | KLM-1.6 | 33.95 | tinyLM-8M-exp (6.82), Rambley-V2-Delta-10M (6.70), ZeroS-Micro-v1.0 (6.49), Atom 3.4M (6.37) |

## Embryo-56M detail

| Task | n | acc | acc_norm | ± stderr (acc_norm) | chance | (acc_norm − chance) / stderr |
|---|---|---|---|---|---|---|
| arc_easy | 2376 | 35.61 | 32.95 | 0.96 | 25 | +8.2 |
| arc_challenge | 1172 | 17.75 | 21.42 | 1.20 | 25 | -3.0 |
| hellaswag | 10042 | 26.50 | 26.83 | 0.44 | 25 | +4.1 |
| piqa | 1838 | 56.58 | 54.30 | 1.16 | 50 | +3.7 |
| arithmark-3 | 1000 | 36.50 | 36.40 | 1.52 | 25 | +7.5 |

## Side check, not in the index: raw text vs the cmf-im-v1 chat frame

Chat frame: the task prompt as the user turn, each choice as the assistant answer.

| Task | raw acc / acc_norm | chat acc / acc_norm | ± stderr (chat acc_norm) |
|---|---|---|---|
| arc_easy | 35.61 / 32.95 | 32.28 / 31.78 | ±0.96 |
| arc_challenge | 17.75 / 21.42 | 16.55 / 22.18 | ±1.21 |
| piqa | 56.58 / 54.30 | 55.98 / 55.01 | ±1.16 |
