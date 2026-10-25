"""SAM3 VIDEO reference fixtures for cortiq parity / e2e tests.

Generates the synthetic tracking clip (white background + a red square that
MOVES between frames + a static blue stripe), runs the reference
`Sam3VideoModel` (transformers 5.20.2, facebook/sam3 weights) over it and
dumps the tracker's intermediate and final outputs:

  frames/frame_00.png ..            the clip (384x384 PNG, byte-identical to
                                    what the Rust e2e test synthesizes)
  gt.json                           per-frame final object_ids / scores / boxes
  f<i>/...                          intermediates for the first DUMP_FRAMES
    det_boxes.f32  det_scores.f32  det_masks.f32     detector output (N,4)/(N,)/(N,288*288)
    neck_l2.f32                    tracker neck level-2 feature map (72*72,1,256)
    pred_masks.f32                 propagated mask logits per object (N_obj,288*288)
    obj_score_logits.f32           (N_obj,)
    object_pointer.f32             (N_obj*?,256)
    maskmem_features.f32           per object, first object only (HW,1,64)
    out_masks.f32  out_scores.f32  out_ids.json     build_outputs() result

Usage:  HF_TOKEN=... python3 tools/sam3_video_ref.py <out_dir>
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIZE = 384
N_FRAMES = 8
SQUARE = 128
STEP_X = 16
STEP_Y = 8
PROMPT = "red square"
DUMP_FRAMES = 3

REPO = "facebook/sam3"
LOCAL_MODEL = "/home/hermes/models/sam3-hf"


def square_box(i):
    """Ground-truth xyxy of the moving red square in frame i."""
    x0 = 64 + STEP_X * i
    y0 = 64 + STEP_Y * i
    return [x0, y0, x0 + SQUARE, y0 + SQUARE]


def make_frame(i):
    """Byte-for-byte the same recipe as the Rust e2e test uses."""
    img = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    img[300:340, :, :] = np.array([30, 60, 200], dtype=np.uint8)
    x0, y0, x1, y1 = square_box(i)
    img[y0:y1, x0:x1, :] = np.array([200, 30, 30], dtype=np.uint8)
    return img


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "crates/cortiq-engine/tests/fixtures/sam3_video")
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(N_FRAMES):
        arr = make_frame(i)
        Image.fromarray(arr).save(frames_dir / f"frame_{i:02d}.png")
        frames.append(Image.fromarray(arr))

    from transformers import Sam3VideoModel, Sam3VideoProcessor

    tok = os.environ.get("HF_TOKEN")
    processor = Sam3VideoProcessor.from_pretrained(REPO, token=tok)
    model = Sam3VideoModel.from_pretrained(LOCAL_MODEL, dtype=torch.float32).eval()

    # uint8 [T,C,H,W] 0..255 — the video processor does its own rescale/resize
    video = torch.from_numpy(np.stack([np.array(f) for f in frames])).permute(0, 3, 1, 2).contiguous()
    session = processor.init_video_session(video=video)
    processor.add_text_prompt(session, PROMPT)

    gt = {"prompt": PROMPT, "size": [SIZE, SIZE], "frames": []}
    for i in range(N_FRAMES):
        out_model = model(inference_session=session, frame_idx=i)
        post = processor.postprocess_outputs(session, out_model, original_sizes=[[SIZE, SIZE]])
        gt["frames"].append(
            {
                "frame": i,
                "object_ids": [int(v) for v in post["object_ids"]],
                "scores": [float(v) for v in post["scores"]],
                "boxes": [[float(v) for v in b] for b in post["boxes"]],
                "gt_square": square_box(i),
            }
        )
        if i >= DUMP_FRAMES:
            continue
        d = out / f"f{i}"
        d.mkdir(parents=True, exist_ok=True)

        def f32(rel, t):
            path = d / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            t.detach().cpu().float().numpy().astype(np.float32).tofile(path)
        print(f"[ref] frame {i}: objects={gt['frames'][-1]['object_ids']} "
              f"scores={[round(s,4) for s in gt['frames'][-1]['scores']]} "
              f"boxes={[[round(v,1) for v in b] for b in gt['frames'][-1]['boxes']]}", flush=True)

        # ── tracker neck features cached by this frame's forward ───────
        cached = session.cache.get_vision_features(i)
        assert cached is not None, f"no cached vision features for frame {i}"
        neck_feats = cached["vision_feats"]  # 3 levels, (HW,1,C)
        f32("neck_l2.f32", neck_feats[-1])

        od = session.output_dict_per_obj
        for obj_idx in sorted(od):
            odx = od[obj_idx]
            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                if i in odx[key]:
                    cur = odx[key][i]
                    pre = f"obj{obj_idx}/{key}"

                    def flat(t):
                        while t.dim() > 1 and t.shape[0] == 1:
                            t = t.squeeze(0)
                        return t

                    f32(pre + "/pred_masks.f32", flat(cur["pred_masks"].reshape(-1, *cur["pred_masks"].shape[-2:]) if cur["pred_masks"].dim() == 4 else cur["pred_masks"]))
                    for k in ("object_score_logits", "object_pointer"):
                        if k in cur:
                            f32(f"{pre}/{k}.f32", cur[k].reshape(-1))
                    for k in ("maskmem_features", "maskmem_pos_enc"):
                        if k in cur:
                            f32(f"{pre}/{k}.f32", cur[k].reshape(-1, cur[k].shape[-1]))
                    if "high_res_masks" in cur:
                        hm = cur["high_res_masks"]
                        f32(pre + "/high_res_masks.f32", hm.reshape(-1, *hm.shape[-2:]))
                    print(f"[ref]   obj{obj_idx} {key}: pred_masks{tuple(cur['pred_masks'].shape)} "
                          f"maskmem={None if 'maskmem_features' not in cur else tuple(cur['maskmem_features'].shape)} "
                          f"ptr={None if 'object_pointer' not in cur else tuple(cur['object_pointer'].shape)}", flush=True)

        # final per-frame outputs (build_outputs)
        ids = sorted(out_model.obj_id_to_mask)
        if ids:
            f32("out_masks.f32", torch.cat([out_model.obj_id_to_mask[k] for k in ids]).reshape(len(ids), -1))
            (d / "out_ids.json").write_text(json.dumps(ids))
        f32("out_scores.f32", torch.tensor([out_model.obj_id_to_score.get(k, 0.0) for k in ids]))

    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    print(f"[ref] wrote {out}", flush=True)


if __name__ == "__main__":
    main()