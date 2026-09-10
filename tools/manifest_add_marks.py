#!/usr/bin/env python3
"""Insert shard checkpoints into a CmfStreamWriter manifest after the fact.

    python3 tools/manifest_add_marks.py out.cmf.manifest model.safetensors.index.json

`cortiq convert --resume` skips source shards the manifest marks as done.
A manifest written by a build from before marks existed has none, so a
resume discards everything and the run starts over. This reconstructs them.

It is sound because of how the converter writes: shards are consumed in the
index's order and every payload is appended, so the manifest's tensor order
IS the shard order. For each shard, in order, the last of its tensors ends
at a byte offset that is exactly the checkpoint the converter would have
recorded — provided every tensor that shard produces is present. A shard
missing any of its outputs was interrupted; that one and everything after
it are left unmarked, so the resume redoes them.

The mapping from an output tensor back to its shard is the index's
weight_map keyed by the SOURCE name, so the converter's renames have to be
undone — hence the deepseek_v4 rules below. Other architectures pass their
names through unchanged and need nothing.
"""
import json, sys, re, os


def source_names(canon: str):
    """Every source spelling that could have produced this output name."""
    yield canon
    c = canon
    if c == "model.embed_tokens.weight":
        yield "embed.weight"
    if c == "lm_head.weight":
        yield "head.weight"
    if c == "model.norm.weight":
        yield "norm.weight"
    if c.startswith("model.hc_head_"):
        yield c[len("model."):]
    m = re.match(r"model\.layers\.(\d+)\.(.*)", c)
    if m:
        li, tail = m.group(1), m.group(2)
        t = tail
        t = t.replace("input_layernorm.weight", "attn_norm.weight")
        t = t.replace("post_attention_layernorm.weight", "ffn_norm.weight")
        t = t.replace("mlp.expert_bias", "ffn.gate.bias")
        t = t.replace("mlp.tid2eid", "ffn.gate.tid2eid")
        t = t.replace("mlp.gate.weight", "ffn.gate.weight")
        t = t.replace("mlp.shared_expert.", "ffn.shared_experts.")
        t = t.replace("mlp.experts.", "ffn.experts.")
        t = (t.replace(".gate_proj.weight", ".w1.weight")
              .replace(".up_proj.weight", ".w3.weight")
              .replace(".down_proj.weight", ".w2.weight"))
        t = t.replace("self_attn.", "attn.")
        yield f"layers.{li}.{t}"


def main():
    man_path, index_path = sys.argv[1], sys.argv[2]
    weight_map = json.load(open(index_path))["weight_map"]
    # shard -> the source tensors it holds, and the order shards appear in
    per_shard, order = {}, []
    for name, shard in weight_map.items():
        if shard not in per_shard:
            per_shard[shard] = set()
            order.append(shard)
        per_shard[shard].add(name)
    order.sort()

    head, entries = None, []
    for line in open(man_path):
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            break          # a truncated last line: the process was killed
        if "data_off" in v:
            head = line
        elif "mark" in v:
            print("манифест уже содержит пометки — ничего не делаю")
            return 1
        else:
            entries.append(v)
    if head is None:
        print("нет заголовка data_off — это не манифест")
        return 1

    # Which shard produced each written tensor. Scales are consumed with
    # their weight and never reach the output, so they are not expected.
    owner = {}
    for shard, names in per_shard.items():
        for n in names:
            owner[n] = shard
    got = {}
    for i, e in enumerate(entries):
        for cand in source_names(e["name"]):
            if cand in owner:
                got.setdefault(owner[cand], []).append(i)
                break

    out, done = [head], 0
    cursor_of = lambda i: entries[i]["off"] + entries[i]["nbytes"]
    written = 0
    for shard in order:
        expected = {n for n in per_shard[shard] if not n.endswith(".scale")}
        idxs = got.get(shard, [])
        produced = set()
        for i in idxs:
            for cand in source_names(entries[i]["name"]):
                if cand in expected:
                    produced.add(cand)
                    break
        if not idxs or produced != expected:
            break                       # interrupted here; stop marking
        last = max(idxs)
        while written <= last:
            out.append(json.dumps(entries[written], separators=(",", ":")))
            written += 1
        out.append(json.dumps({"mark": shard, "at": cursor_of(last)},
                              separators=(",", ":")))
        done += 1

    if done == 0:
        print("ни один шард не завершён целиком — восстанавливать нечего")
        return 1
    backup = man_path + ".bak"
    os.replace(man_path, backup)
    with open(man_path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"пометок вставлено: {done} из {len(order)} шардов; "
          f"тензоров сохранено: {written} из {len(entries)}; "
          f"прежний манифест: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
