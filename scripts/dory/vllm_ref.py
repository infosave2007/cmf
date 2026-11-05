#!/usr/bin/env python3
"""The vendor's own code (vLLM PR #60149) on the reference prompts, to check
that ref_dory.py transcribes it: greedy tokens and top logprobs at float32."""
import json, sys
from vllm import LLM, SamplingParams

model, prompts, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {}
P = json.load(open(prompts))
for loops, mode in ((2, "per_loop"), (1, "per_loop"), (4, "per_loop"), (2, "last_loop")):
    llm = LLM(
        model=model, dtype="float32", max_model_len=4096, gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        # The Triton attention kernel asks for more shared memory than an L4 has.
        compilation_config=0,
        attention_config={"backend": "FLEX_ATTENTION"},
        hf_overrides={"n_recurrent_loops": loops, "recurrent_kv_cache_mode": mode},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5, stop_token_ids=[2, 11])
    outs = llm.generate([{"prompt_token_ids": p["ids"]} for p in P], sp)
    for p, o in zip(P, outs):
        c = o.outputs[0]
        lp0 = {int(k): v.logprob for k, v in c.logprobs[0].items()}
        res.setdefault(p["name"], {})[f"l{loops}_{mode}"] = {"greedy": list(c.token_ids), "lp0": lp0}
        print(p["name"], loops, mode, list(c.token_ids)[:16], flush=True)
    json.dump(res, open(out, "w"))
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
