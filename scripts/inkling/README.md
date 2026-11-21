# Inkling-2-Mini scripts

How the numbers of the [Inkling-2-Mini-cmf](https://huggingface.co/infosave/Inkling-2-Mini-cmf) page were made.

| script | what it does |
|---|---|
| `calib_texts.py` | the calibration texts of the dense trunk's GPTQ (prose, Rust, Python outside the evaluation texts) and the two commands that make the published file |
| `ref_layers.py` | the f32 reference: the release's layers (transformers PR #51050) one application at a time, experts decoded from NVFP4 / MXFP8; perplexity over the windows `cortiq ppl` scores |
| `bench2.sh` | `cortiq bench --core` decode and prompts by length on one backend, with the card's peak memory |
| `bench_rows.py` | bench2's lines as the rows the figures read |
| `card_figures.py` | the page's figures, EN / RU / ZH (`rsvg-convert` on PATH) |

Perplexity: `cortiq ppl FILE --file wiki.test.raw --windows 12 --window-len 512` (and `--windows 2 --window-len 4096`),
the same windows `ref_layers.py` scores.
