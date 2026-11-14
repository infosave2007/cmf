# Reference for cortiq's Qwen3-TTS port: qwen_tts 0.1.1 on the local HF
# snapshot in float32 on the CPU, greedy codes (do_sample off on both levels,
# the release's repetition penalty kept), the token ids, the codes and the
# decoded waveform.
import json, sys, time
import numpy as np, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

hf, out, text, lang, spk = sys.argv[1:6]
torch.manual_seed(0)
m = Qwen3TTSModel.from_pretrained(hf, device_map="cpu", dtype=torch.float32)
ids = m._tokenize_texts([m._build_assistant_text(text)])
t0 = time.time()
codes, _ = m.model.generate(input_ids=ids, instruct_ids=[None], languages=[lang], speakers=[spk],
                            non_streaming_mode=True, do_sample=False, subtalker_dosample=False,
                            top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05, max_new_tokens=600)
t1 = time.time()
wavs, fs = m.model.speech_tokenizer.decode([{"audio_codes": codes[0]}])
t2 = time.time()
w = np.asarray(wavs[0], dtype=np.float32)
json.dump({"ids": ids[0][0].tolist(), "codes": codes[0].tolist()}, open(f"{out}/ref.json", "w"))
w.tofile(f"{out}/ref_wav.f32")
sf.write(f"{out}/ref.wav", w, fs)
print(f"ref: {len(codes[0])} frames, {len(w)/fs:.2f} s, generate {t1-t0:.1f} s, decode {t2-t1:.1f} s, ids {ids[0][0].tolist()[:6]}..")
