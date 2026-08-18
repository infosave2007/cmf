# Examples

Every file here was produced by `cortiq ltx-video` reading
[`ltx25-q4tp.cmf`](../ltx25-q4tp.cmf) — one Rust process, no PyTorch, no
Python in the render path. The `.mp4` files carry **the soundtrack the same
48 transformer blocks generated alongside the picture**; the `.gif` files are
silent previews for the model card.

| clip | seed | prompt |
|---|---|---|
| `corgi` | 42 | A corgi in a chef hat flips a pancake in a sunlit kitchen. Warm morning light, static camera. |
| `neon` | 7 | Neon rain on a Tokyo side street at night, a lone figure with a translucent umbrella walks past ramen shop signs, reflections rippling in the puddles, slow dolly. |
| `anvil` | 3 | A blacksmith hammers glowing steel on an anvil, sparks flying, the ring of metal echoing in the workshop. |
| `whale` | 11 | A humpback whale glides through a shaft of sunlight in deep blue water, plankton drifting like dust, the camera rises with it toward the surface. |
| `glass` | 23 | Molten glass is blown into a bulb over an orange furnace, the glowing gather stretching and rotating, sparks drifting in the dark workshop. |
| `market` | 5 | A spice market at golden hour, saffron and paprika in open sacks, a vendor scoops turmeric and the dust catches the light, handheld camera. |

All six are 384×256, 49 frames at 24 fps, eight ancestral Euler steps:

```bash
cortiq ltx-video --model ltx25-q4tp.cmf --seed 42 \
  --height 256 --width 384 --frames 49 --fps 24 \
  --prompt "A corgi in a chef hat flips a pancake in a sunlit kitchen. Warm morning light, static camera." \
  --out-dir corgi/ --out-audio corgi.wav

ffmpeg -framerate 24 -i corgi/frame_%04d.ppm -i corgi.wav \
  -pix_fmt yuv420p -c:v libx264 -crf 18 -c:a aac -b:a 192k -shortest corgi.mp4
```

The `.wav` files are the raw 48 kHz stereo the audio VAE and its vocoder
produced, before muxing.

## Two stages, 768×512

`hq1.mp4` is the same corgi prompt at 768×512 with `--two-stage`: eight
ancestral steps at half resolution, the learned ×2 latent upscaler, then
three deterministic steps at full resolution.

```bash
cortiq ltx-video --model ltx25-q4tp.cmf --two-stage --seed 42 \
  --height 512 --width 768 --frames 49 \
  --prompt "A corgi in a chef hat flips a pancake in a sunlit kitchen. Warm morning light, static camera." \
  --out-dir hq/ 
```

`ref1.mp4` is the **reference implementation's** output for the same prompt,
seed and settings — PyTorch, fp8-cast weights, an NVIDIA card. It is not the
same clip, and it is not supposed to be: the ancestral sampler draws its noise
from a different generator, so the two are different samples of the same
model rather than a reproduction. Put them side by side to judge the port on
what it should be judged on — whether it renders the same *kind* of thing at
the same quality.
