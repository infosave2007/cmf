"""LFM3-Audio reference in float32 (Transformers, the checkpoint's own code):
for each WAV, the processor's log-mel features, the audio encoder's output,
the projected rows the language model reads, the prompt ids and a greedy
transcript; then a greedy TTS frame sequence with the MaskGIT order fixed to
codebook order (`torch.randperm` patched to `arange`) and its codec
waveform. Everything lands in OUT/ as .npy / .json for audio_cmp.py.

usage: python3 -I lfm3_audio_ref.py MODEL_DIR OUT a.wav [b.wav ...]
"""

import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    wavs = sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, dtype=torch.float32, device_map=dev, trust_remote_code=True
    ).eval()
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    inner = model.model
    enc_out = {}
    inner.audio_encoder.register_forward_hook(lambda m, i, o: enc_out.__setitem__("enc", o[0].detach()))
    summary = []
    for i, wav in enumerate(wavs):
        conv = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "audio": wav}]},
        ]
        inputs = proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(dev)
        feats = inputs["input_features"].float().cpu().numpy()
        np.save(f"{out}/clip{i}_mel.npy", feats[0])
        with torch.no_grad():
            rows = inner.get_audio_features(inputs["input_features"], inputs["feature_attention_mask"])
            np.save(f"{out}/clip{i}_enc.npy", enc_out["enc"][0].float().cpu().numpy())
            np.save(f"{out}/clip{i}_rows.npy", rows.float().cpu().numpy())
            gen = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                suppress_mixed=True,
                suppress_audio_out=True,
                return_dict_in_generate=True,
            )
        n0 = inputs["input_ids"].shape[1]
        ids = gen.sequences[0, n0:].tolist()
        text = proc.decode(ids, skip_special_tokens=True)
        rec = {
            "wav": os.path.basename(wav),
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "mel_frames": int(inputs["feature_attention_mask"].sum()),
            "rows": list(rows.shape),
            "ids": ids,
            "text": text,
        }
        summary.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "prompt_ids"}), flush=True)

    # Greedy TTS with the codebooks revealed in order 0..11.
    real_randperm = torch.randperm
    torch.randperm = lambda n, **kw: torch.arange(n, device=kw.get("device"))
    try:
        conv = [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
        inputs = proc.apply_chat_template(
            conv, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=256, audio_temperature=0.0, return_dict_in_generate=True
            )
            n = int(gen.audio_lengths[0])
            codes = gen.audio_codes[:1, :, :n]
            codec = proc.audio_tokenizer.to(device=dev).eval()
            wave = codec.decode(codes).audio_values[0].float().cpu().numpy()
        np.save(f"{out}/tts_codes.npy", codes[0].cpu().numpy())
        sf.write(f"{out}/tts.wav", wave.reshape(-1), 24_000)
        summary.append({"tts_prompt_ids": inputs["input_ids"][0].tolist(), "frames": n,
                        "seconds": round(wave.size / 24_000, 2)})
        print("tts frames", n, "codes", tuple(codes.shape), "seconds", round(wave.size / 24_000, 2), flush=True)
    finally:
        torch.randperm = real_randperm
    json.dump(summary, open(f"{out}/ref.json", "w"), indent=1)


if __name__ == "__main__":
    main()
