"""LFM3-Audio speech parity: greedy TTS with the codebooks revealed in
order (CMF_LFM3_MASKGIT_FIXED=1, the reference's patched randperm) through
`cortiq run --speech-out`, frames compared with lfm3_audio_ref.py's
tts_codes.npy, and both waveforms' lengths and RMS.

usage: python3 -I tts_cmp.py REF_DIR MODEL.cmf [CORTIQ]
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def main():
    ref_dir, model = sys.argv[1], sys.argv[2]
    cortiq = sys.argv[3] if len(sys.argv) > 3 else "/root/lfm3/bin/cortiq"
    want = np.load(os.path.join(ref_dir, "tts_codes.npy"))  # [12, frames]
    with tempfile.TemporaryDirectory() as tmp:
        wav, codes = os.path.join(tmp, "tts.wav"), os.path.join(tmp, "codes.json")
        env = {**os.environ, "CMF_LFM3_MASKGIT_FIXED": "1", "CMF_LFM3_CODES_DUMP": codes, "XDG_RUNTIME_DIR": "/tmp"}
        out = subprocess.run(
            [cortiq, "run", model, "--system", "Perform TTS in American English using a female voice.",
             "--prompt", "The quick brown fox jumps over the lazy dog.", "--speech-out", wav, "--greedy",
             "--max-tokens", "256"],
            capture_output=True, text=True, env=env)
        print(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "", out.stdout.strip())
        if out.returncode:
            print("FAILED:", out.stderr[-1500:])
            return
        got = np.array(json.load(open(codes)), dtype=np.int64).reshape(-1, 12).T
        audio, _ = sf.read(wav)
        # Intelligibility: the speech back through the model's own ASR.
        asr = subprocess.run(
            [cortiq, "run", model, "--system", "Perform ASR in English.", "--audio", wav, "--prompt", "",
             "--greedy", "--max-tokens", "64"],
            capture_output=True, text=True, env={**os.environ, "XDG_RUNTIME_DIR": "/tmp"})
        heard = [l for l in asr.stdout.splitlines() if l.strip() and not l.startswith(("Loading", "sampling", "Ready", "Prompt", "["))]
        print("ASR of the speech:", repr(" ".join(heard).strip()))
    ref_audio, _ = sf.read(os.path.join(ref_dir, "tts.wav"))
    n = min(got.shape[1], want.shape[1])
    same = (got[:, :n] == want[:, :n])
    first = next((f for f in range(n) if not same[:, f].all()), None)
    print(f"frames: cortiq {got.shape[1]} vs ref {want.shape[1]}; identical codes {same.mean() * 100:.1f} %"
          f" of the common {n}; first differing frame {first}")
    print(f"audio: cortiq {len(audio) / 24000:.2f} s rms {np.sqrt((audio ** 2).mean()):.4f}; "
          f"ref {len(ref_audio) / 24000:.2f} s rms {np.sqrt((ref_audio ** 2).mean()):.4f}")
    if got.shape == want.shape and (got == want).all():
        m = min(len(audio), len(ref_audio))
        print(f"waveform max |d| {np.abs(audio[:m] - ref_audio[:m]).max():.2e} (16-bit WAV)")


if __name__ == "__main__":
    main()
