# Market skill: next-hour direction and range (BTCUSDT, EURUSD, USDJPY)

A `cortiq skill bake` experiment on Qwen3.5-0.8B (q8_2f) with the fixed bake (`--through-attn`, soft mask), run on an
RTX 3060 with CUDA. It repeats the protocol of `examples/btc_market_skill` and of the article draft
`docs/HABR_QWEN_BTC_SKILL.ru.md`, whose 47.75 % → 47.25 % had been baked with the mask gradient skipping the frozen
attention and a hard τ = 0.5 cut. It also adds two FX pairs and a second target. This measures classification accuracy
only. It is not a trading strategy.

- Data: 1h bars from 2024-02-07 to 2026-11-24 (Binance BTCUSDT, Yahoo EURUSD=X and USDJPY=X).
- Features and prompt: the article's causal features of 48 closed bars; FX prompts have no volume fields.
- Splits: per market, chronological 70/15/15 with a 49 h purge; the next bar must be exactly +1 h.
- Targets:
  - `dir`: UP if `close[t+1] > close[t]`, else DOWN (the article's target).
  - `range`: HIGH if the next bar's high-low range is above the market's median over its own train split, else LOW.
- Training: one skill per target on all three markets (4000 balanced records each).
- Checkpoint: chosen by validation accuracy inside the bake.
- Evaluation: through `serve` with the skill overlaid (`skill export`, `skill apply --overlay`, `serve --skill`),
  `class_tokens`, thinking off.
- Test period: 2026-06-24 to 2026-11-23, 1000 evenly spaced hours a market, opened once after both skills were frozen.

`PROTOCOL.json` was frozen before any prediction. `JOURNAL.ru.md` is the run log.

## Test period (3000 hours)

| target | base: acc / balanced | skill: acc / balanced | skill − base, acc (95 % CI, paired block bootstrap) | skill, 20 % most confident |
|---|---:|---:|---:|---:|
| `dir` | 51.2 / 50.0 | 50.8 / 50.8 | −0.4 (−2.7, +1.9) | 53.5 |
| `range` | 66.1 / 50.0 | 75.6 / 70.7 | +9.5 (+7.4, +11.7) | 92.2 |

`range` by market (balanced accuracy), with the classical baselines on the same hours:

| market | base | skill | persistence (current range) | logistic regression (features + hour) |
|---|---:|---:|---:|---:|
| BTCUSDT | 50.0 | 70.3 | 71.5 | 69.4 |
| EURUSD | 50.0 | 69.8 | 67.0 | 72.8 |
| USDJPY | 50.0 | 71.6 | 71.5 | 72.3 |

- **Direction.** Next-hour direction stays at a coin flip. So does every honest baseline (majority, momentum,
  reversal, logistic regression: 46–54 %). Fixing the bake did not create a signal that is not in the data.
- **Range.** The skill learns next-hour volatility from the prompt text — the base answers LOW every time. It reaches the
  level of a logistic regression on the same features, not beyond it.
- **Class mix.** The later period is calmer: 63–72 % of test hours are LOW against 50 % in train. Read the balanced
  accuracy.

## Reproduce

```sh
python3 market_skill.py fetch-binance BTCUSDT "2024-02-07 00:00" "2026-11-24 00:00" data/BTCUSDT_1h.csv
python3 market_skill.py fetch-yahoo EURUSD=X data/EURUSD_1h.csv
python3 market_skill.py fetch-yahoo USDJPY=X data/USDJPY_1h.csv
python3 market_skill.py prepare data/prep data/BTCUSDT_1h.csv:BTCUSDT data/EURUSD_1h.csv:EURUSD data/USDJPY_1h.csv:USDJPY
./bake.sh dir dir1 && ./bake.sh range range1      # 17 min each on an RTX 3060 (CMF_GPU=cuda)
./run_eval.sh                                       # overlays, held-set parity, validation and test
python3 show_report.py results test
```

Yahoo's 1h history reaches back 730 days, so a later fetch covers a later period. The scripts use the standard
library only. `bake.sh` and the `run_*` scripts expect the q8_2f base at `~/skill-ode/cmf/qwen35-0.8b-q8_2f.cmf` and
the CLI at `~/cmf-skillode/target/release/cortiq`.
