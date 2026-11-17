# Haru reference (higurashi-ai/Haru's PyTorch package, float32 on the CPU):
#   codec  IN OUT MEANS     the codec round trip (latent means + audio)
#   talk   IN OUT REF SEED  a conversation as haru-inference runs it (text out)
import sys, json, time
import numpy as np, torch, soundfile as sf
from haru.audio import load_audio, prepare_reference
from haru.models import load_model

mode = sys.argv[1]
model = load_model("/root/haru/hf", device="cpu", dtype=torch.float32)
fs = model.config.frame_samples
if mode == "codec":
    _, _, inp, out, means_path = sys.argv
    a = load_audio(inp)
    a = np.pad(a, (0, -len(a) % fs))
    m = model.codec.encode(torch.from_numpy(a))
    m.float().numpy().tofile(means_path)
    w = model.codec.decode(m)[0].float().clamp(-1, 1).numpy()
    sf.write(out, w, 24000)
    print(f"ref codec: {m.shape[1]} frames")
else:
    _, _, inp, out, refv, seed = sys.argv
    from haru.session import DuplexConfig, DuplexSession
    user = load_audio(inp)
    count = max(1, -(-len(user) // fs))
    user = np.pad(user, (0, count * fs - len(user)))
    reference = prepare_reference(load_audio(refv))
    cfg = DuplexConfig(max_frames=count + 1, seed=int(seed))
    s = DuplexSession(model, "", reference, cfg)
    t0 = time.time()
    frames = [s.start()]
    for i in range(count - 1):
        frames.append(s.step(user[i * fs:(i + 1) * fs]))
    reply = np.concatenate([f.audio for f in frames])
    sf.write(out, reply, 24000)
    print(f"ref talk: {count} frames in {time.time() - t0:.1f} s; text: {''.join(f.text for f in frames).strip()!r}")
