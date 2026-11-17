# Skills on CUDA: measurement scripts

The runs behind the CHANGELOG's skill entries (`--skill`, `--blend`, `--route-dynamic` on the CUDA hybrid arm,
`skill bake` on CUDA), on a RunPod A40 under `/root/h3i`. Every GPU run takes `flock /root/h3i/gpu.lock`.

- `pod/t1.sh`: greedy text of one file with and without a skill, CPU against CUDA.
- `pod/parity.sh` + `pod/parity_cmp.py`: 64 greedy tokens of four prompts on CUDA (ids and every step's
  logits kept), the float host teacher-forced along them and free; top-1 agreement, logit gap, where the
  free texts part.
- `pod/speed.sh`: `bench` with and without a skill, interleaved: decode (drafts on and off) and a 4096-token
  prompt.
- `pod/prep.py`, `pod/bake.sh`, `pod/serve_eval.sh`, `pod/eval.py`, `pod/run4b.sh`: the emotion skill
  (dair-ai/emotion): corpus, DTG-MA bake (answer span), greedy accuracy of a server on the 2000 test texts.
- `prep_code.py`, `pod/cbake.sh`, `pod/cbake2.sh`: the coding skill from a Cortiq-Coder dataset (all-token LM
  and answer-span bakes); `pod/run4b_code.sh` runs it on the 4B (Phase A, then FCD on the best checkpoint).
- `pod/run4b_more.sh`: 500 more Phase-A steps of the 4B emotion mask from its checkpoint, export, overlay and
  the 2000-text accuracy; `pod/run4b_fcd.sh`: FCD on that mask with lr 1e-6 (2 layers, 2 steps), then 500 more Phase-A steps;
  `pod/run08_fcd.sh`: FCD with lr 1e-6 on the 0.8B emotion mask.
