"""LFM3-Omni reference in float32 (Transformers, the checkpoint's own code):
a greedy one-sentence description of each image, a greedy transcript of each
WAV (with the audio encoder's projected rows), and a greedy TTS frame
sequence with the MaskGIT order fixed to codebook order (`torch.randperm`
patched to `arange`) and its codec waveform. Everything lands in OUT/ for
omni_cmp.py.

usage: python3 -I lfm3_omni_ref.py MODEL_DIR OUT --image a.jpg [...] --wav a.wav [...]
"""

import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main():
    args = sys.argv[1:]
    model_dir, out = args[0], args[1]
    images, wavs, cur = [], [], None
    for a in args[2:]:
        if a in ("--image", "--wav"):
            cur = a
        elif cur == "--image":
            images.append(a)
        else:
            wavs.append(a)
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, dtype=torch.float32, device_map=dev, trust_remote_code=True
    ).eval()
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    summary = []

    def text_turn(conv, max_new):
        inputs = proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new, generation_mode="text")
        n0 = inputs["input_ids"].shape[1]
        ids = gen.sequences[0, n0 : gen.sequence_lengths[0]].tolist()
        return inputs, ids, proc.decode(ids, skip_special_tokens=True)

    for i, img in enumerate(images):
        conv = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "Describe this image in one sentence."}]}]
        inputs, ids, text = text_turn(conv, 48)
        rec = {"image": os.path.basename(img), "prompt_len": inputs["input_ids"].shape[1], "ids": ids, "text": text}
        summary.append(rec)
        print(json.dumps(rec), flush=True)

    for i, wav in enumerate(wavs):
        conv = [{"role": "system", "content": "Perform ASR in English."},
                {"role": "user", "content": [{"type": "audio", "audio": wav}]}]
        inputs, ids, text = text_turn(conv, 96)
        with torch.no_grad():
            rows = model.model.get_audio_features(inputs["input_features"], inputs["feature_attention_mask"])
        np.save(f"{out}/clip{i}_rows.npy", rows.float().cpu().numpy())
        rec = {"wav": os.path.basename(wav), "prompt_len": inputs["input_ids"].shape[1], "ids": ids, "text": text}
        summary.append(rec)
        print(json.dumps(rec), flush=True)

    real_randperm = torch.randperm
    torch.randperm = lambda n, **kw: torch.arange(n, device=kw.get("device"))
    try:
        conv = [{"role": "system", "content": "Perform TTS in American English using a female voice."},
                {"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]
        inputs = proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, generation_mode="audio", audio_temperature=0.0)
            n = int(gen.audio_lengths[0])
            codes = gen.audio_codes[:1, :, :n]
            codec = proc.audio_tokenizer.to(device=dev).eval()
            wave = codec.decode(codes).audio_values[0].float().cpu().numpy()
        np.save(f"{out}/tts_codes.npy", codes[0].cpu().numpy())
        sf.write(f"{out}/tts.wav", wave.reshape(-1), 24_000)
        summary.append({"tts_prompt_ids": inputs["input_ids"][0].tolist(), "frames": n,
                        "seconds": round(wave.size / 24_000, 2)})
        print("tts frames", n, "seconds", round(wave.size / 24_000, 2), flush=True)
    finally:
        torch.randperm = real_randperm
    json.dump(summary, open(f"{out}/ref.json", "w"), indent=1)


if __name__ == "__main__":
    main()
