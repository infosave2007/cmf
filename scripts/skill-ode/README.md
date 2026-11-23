# The patent-2 ODE skill: a fair re-check

The DTG-MA notebook `prune_diff_eq_06b.ipynb` reported 3.3 % → 100 % for Qwen3-0.6B on templated differential
equations. Its base model got a raw prompt and 32 tokens and its test came from the training generator. These
scripts measure it with a held-out coefficient set, three base settings, and the same recipe in `cortiq skill bake`
(numbers in the CHANGELOG and docs/SKILLS.md, "The gradient through attention").

- `ode_repro.py`: PyTorch (MPS/CUDA/CPU) — the base three ways (raw 32 tokens, chat template with 300 tokens,
  one-shot), then the notebook's mask on every Linear (`--variant all`) or on the FFN neurons only (`--variant ffn`).
- `make_corpus.py`: the same 300 training equations and 40 held ones as `train.txt` / `held.txt`.
- `bake.sh`, `serve_eval.sh` + `engine_eval.py`: the engine bake and its accuracy over `/v1/completions`.

```sh
python3 ode_repro.py --variant ffn --out B_ffn.json
python3 make_corpus.py
cortiq convert --model Qwen/Qwen3-0.6B --quant f16 --output cmf/qwen3-0.6b-f16.cmf
./bake.sh k4 --steps-a 510 --eval-every 34 --steps-b 0 --batch 4 --mask-init 2.0 --lr-a 0.1 \
  --l1-init 0.56 --l1-step 0 --tau 0.001 --through-attn
./serve_eval.sh k4 runs/k4/specialist.cmf
```
