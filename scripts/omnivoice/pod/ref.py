#!/usr/bin/env python3
"""OmniVoice reference dumps for the cortiq port (float32, deterministic).

For each case: the prompt (input ids, audio mask, target length), the step-0
logits of the conditional and unconditional target rows, the final tokens
[8, T] of a run with position_temperature = class_temperature = 0, the codec's
raw waveform of those tokens and the post-processed one. A voice-clone case
encodes a reference wav (the first case's output) and dumps its codes.

usage: ref.py SRC_DIR OUT_DIR
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
m = OmniVoice.from_pretrained(src, device_map="cuda:0", dtype=torch.float32)
m.eval()

CASES = [
    ("en_auto", "Hello, this is a test of the OmniVoice model running natively in cortiq.", "en", None),
    ("ru_design", "Привет! Это проверка синтеза речи на русском языке.", "ru", "female, low pitch"),
    ("zh_auto", "你好，这是一个语音合成测试。", "zh", None),
    ("en_tag", "I can't believe it [laughter] that is so funny!", "en", "male, british accent"),
]


def run(name, text, lang, instruct, clone=None):
    cfg = OmniVoiceGenerationConfig(position_temperature=0.0, class_temperature=0.0)
    kw = {}
    if clone is not None:
        kw["voice_clone_prompt"] = clone
    task = m._preprocess_all(text=text, language=lang, instruct=instruct, preprocess_prompt=True, **kw)
    inp = m._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0], task.ref_audio_tokens[0],
        task.langs[0], task.instructs[0], cfg.denoise,
    )
    ids = inp["input_ids"][0].cpu().numpy().astype(np.int64)
    am = inp["audio_mask"][0].cpu().numpy().astype(np.uint8)
    np.save(f"{out}/{name}_ids.npy", ids)
    np.save(f"{out}/{name}_audio_mask.npy", am)
    t_len = task.target_lens[0]
    # step-0 logits (cond and uncond target rows), as _generate_iterative builds them
    c_len = ids.shape[1]
    bi = torch.full((2, 8, c_len), m.config.audio_mask_id, dtype=torch.long, device=m.device)
    bm = torch.zeros((2, c_len), dtype=torch.bool, device=m.device)
    ba = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=m.device)
    bi[0] = inp["input_ids"][0]
    bm[0] = inp["audio_mask"][0]
    ba[0, :, :, :] = True
    bi[1, :, :t_len] = inp["input_ids"][0][..., -t_len:]
    bm[1, :t_len] = inp["audio_mask"][0][..., -t_len:]
    ba[1, :, :t_len, :t_len] = True
    pad = torch.arange(t_len, c_len, device=m.device)
    ba[1, :, pad, pad] = True
    with torch.inference_mode():
        lg = m(input_ids=bi, audio_mask=bm, attention_mask=ba).logits.float()
    np.save(f"{out}/{name}_step0_cond.npy", lg[0, :, c_len - t_len:c_len, :].cpu().numpy())
    np.save(f"{out}/{name}_step0_uncond.npy", lg[1, :, :t_len, :].cpu().numpy())
    with torch.inference_mode():
        toks = m._generate_iterative(task, cfg)[0]
    np.save(f"{out}/{name}_tokens.npy", toks.cpu().numpy().astype(np.int64))
    with torch.inference_mode():
        raw = m.audio_tokenizer.decode(toks.unsqueeze(0).to(m.audio_tokenizer.device)).audio_values[0].cpu().numpy()
        post = m._decode_and_post_process(toks, task.ref_rms[0], cfg)
    np.save(f"{out}/{name}_raw.npy", raw.astype(np.float32))
    sf.write(f"{out}/{name}_raw.wav", raw[0], 24000)
    sf.write(f"{out}/{name}.wav", post, 24000)
    meta = {
        "text": text, "lang": task.langs[0], "instruct": task.instructs[0], "target_len": t_len,
        "c_len": c_len, "ref_text": task.ref_texts[0], "ref_rms": task.ref_rms[0],
        "raw_samples": int(raw.shape[-1]), "post_samples": int(post.shape[-1]),
    }
    json.dump(meta, open(f"{out}/{name}.json", "w"), ensure_ascii=False, indent=1)
    print(name, json.dumps(meta, ensure_ascii=False), flush=True)


for c in CASES:
    run(*c)

# voice clone: the first case's output as the reference
vc = m.create_voice_clone_prompt(ref_audio=f"{out}/en_auto.wav", ref_text=CASES[0][1], preprocess_prompt=True)
np.save(f"{out}/clone_ref_codes.npy", vc.ref_audio_tokens.cpu().numpy().astype(np.int64))
json.dump({"ref_text": vc.ref_text, "ref_rms": vc.ref_rms}, open(f"{out}/clone_ref.json", "w"), ensure_ascii=False)
run("en_clone", "This sentence is spoken in the cloned voice of the first sample.", "en", None, clone=vc)
print("REF_DONE", flush=True)
