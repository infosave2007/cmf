---
title: LTX-2.5 on cortiq
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Text to video from one 22 GB CMF file, rendered by Rust
---

# LTX-2.5, rendered by one Rust binary

This Space runs [`cortiq`](https://github.com/infosave2007/cmf) against
[infosave/LTX-2.5-cmf](https://huggingface.co/infosave/LTX-2.5-cmf) — the whole
LTX-2.5 pipeline (a 21 B audio-video diffusion transformer, a Gemma-4 12 B
prompt encoder, a 3-D video VAE and a latent upscaler) in a single
memory-mapped file.

`app.py` collects a prompt and shells out. There is no PyTorch, no diffusers
and no model code in this Space at all.
