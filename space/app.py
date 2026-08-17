"""LTX-2.5 on cortiq — a Gradio front end that only collects a prompt.

Everything numerical happens in one Rust process reading one memory-mapped
CMF file: `cortiq ltx-video`. This file starts it, waits, and hands back the
mp4. There is no torch, no diffusers and no model code here at all.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

REPO = os.environ.get("LTX_CMF_REPO", "infosave/LTX-2.5-cmf")
FILE = os.environ.get("LTX_CMF_FILE", "ltx25-q4tp.cmf")
ASSETS = Path(__file__).parent / "assets"

EXAMPLES = [
    ("corgi", "A corgi in a chef hat flips a pancake in a sunlit kitchen. "
              "Warm morning light, static camera.", 42),
    ("neon", "Neon rain on a Tokyo side street at night, a lone figure with a "
             "translucent umbrella walks past ramen shop signs, reflections "
             "rippling in the puddles, slow dolly.", 7),
    ("whale", "A humpback whale glides through a shaft of sunlight in deep blue "
              "water, plankton drifting like dust, the camera rises with it "
              "toward the surface.", 11),
    ("glass", "Molten glass is blown into a bulb over an orange furnace, the "
              "glowing gather stretching and rotating, sparks drifting in the "
              "dark workshop.", 23),
]

_model_path: str | None = None


def model() -> str:
    """Fetch the container once, then reuse it. 22 GB, so this is slow the
    first time a cold Space is asked for anything."""
    global _model_path
    if _model_path is None:
        _model_path = hf_hub_download(repo_id=REPO, filename=FILE)
    return _model_path


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], capture_output=True
    ).returncode == 0


def render(prompt: str, height: int, width: int, frames: int, seed: int, progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a prompt first.")
    if height % 32 or width % 32:
        raise gr.Error("Height and width must be multiples of 32.")
    if frames % 8 != 1:
        raise gr.Error("Frame count must be 8k+1 (9, 17, 25, 33, 41, 49 …).")

    progress(0.02, desc="fetching the container (22 GB, first run only)")
    path = model()
    work = Path(tempfile.mkdtemp())
    frames_dir = work / "frames"
    cmd = [
        "cortiq", "ltx-video", "--model", path, "--prompt", prompt,
        "--height", str(height), "--width", str(width), "--frames", str(frames),
        "--seed", str(seed), "--out-dir", str(frames_dir),
    ]
    started = time.time()
    log: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    total = 8
    for line in proc.stdout:
        log.append(line.rstrip())
        m = re.search(r"step (\d+)/(\d+)", line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            progress(0.05 + 0.8 * done / total, desc=f"denoising {done}/{total}")
        elif "decoded" in line:
            progress(0.9, desc="video VAE")
    proc.wait()
    if proc.returncode != 0:
        raise gr.Error("render failed:\n" + "\n".join(log[-12:]))

    out = work / "out.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24",
         "-i", str(frames_dir / "frame_%04d.ppm"), "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-crf", "18", str(out)],
        check=True,
    )
    elapsed = time.time() - started
    return str(out), "\n".join(log) + f"\n\ntotal {elapsed:.0f}s"


GPU = has_gpu()
NOTE = (
    "A GPU is attached — expect roughly half a minute per denoising step at "
    "384×256."
    if GPU else
    "**This Space is running on CPU.** The transformer is 21 B parameters, so "
    "a 384×256 clip takes tens of minutes. The sizes below are capped to "
    "something that finishes; for real work run the same binary on your own "
    "machine — the command is identical."
)

with gr.Blocks(title="LTX-2.5 on cortiq", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# LTX-2.5, rendered by one Rust binary\n"
        "Text to video from a single 22 GB [CMF](https://github.com/infosave2007/cmf) "
        "file — no PyTorch, no diffusers, no Python in the render path. "
        "[Model card](https://huggingface.co/infosave/LTX-2.5-cmf)."
    )
    with gr.Tab("Gallery"):
        gr.Markdown("Every clip below came out of `cortiq ltx-video`, 49 frames at 24 fps, 384×256.")
        with gr.Row():
            for name, prompt, seed in EXAMPLES:
                gif = ASSETS / f"{name}.gif"
                if gif.exists():
                    with gr.Column():
                        gr.Image(str(gif), show_label=False, show_download_button=False)
                        gr.Markdown(f"*{prompt}*\n\n`--seed {seed}`")
    with gr.Tab("Generate"):
        gr.Markdown(NOTE)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    value=EXAMPLES[0][1],
                )
                with gr.Row():
                    height = gr.Dropdown([128, 160, 192, 256] if not GPU else [256, 384, 512],
                                         value=128 if not GPU else 256, label="Height")
                    width = gr.Dropdown([192, 256, 320, 384] if not GPU else [384, 512, 768],
                                        value=192 if not GPU else 384, label="Width")
                with gr.Row():
                    frames = gr.Dropdown([9, 17, 25, 33, 41, 49], value=9 if not GPU else 49,
                                         label="Frames")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                go = gr.Button("Render", variant="primary")
            with gr.Column():
                video = gr.Video(label="Result", autoplay=True, loop=True)
                logs = gr.Textbox(label="Renderer output", lines=12)
        go.click(render, [prompt, height, width, frames, seed], [video, logs])
    with gr.Tab("Run it yourself"):
        gr.Markdown(
            "```bash\n"
            "cargo install cortiq-cli\n"
            "hf download infosave/LTX-2.5-cmf ltx25-q4tp.cmf --local-dir .\n"
            "cortiq ltx-video --model ltx25-q4tp.cmf \\\n"
            '  --prompt "A corgi in a chef hat flips a pancake in a sunlit kitchen." \\\n'
            "  --height 256 --width 384 --frames 49 --seed 42 --out corgi.y4m\n"
            "ffmpeg -i corgi.y4m -pix_fmt yuv420p corgi.mp4\n"
            "```\n\n"
            "The GPU is found at run time — Vulkan on Linux and Windows, Metal on "
            "Apple silicon — and everything falls back to the CPU when there is none. "
            "Keep the container on local storage: it is memory-mapped, and a network "
            "filesystem turns every weight into a network round trip."
        )

demo.queue(max_size=8).launch()
